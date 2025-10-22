# agentic_rag_langgraph.py

import os
from typing import TypedDict, List
from dotenv import load_dotenv

# --- LangGraph / LangChain imports (2025 structure) ---
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------
load_dotenv()  # loads .env file if present
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # or "gpt-4.1" if you have access
embeddings = OpenAIEmbeddings()

# ---------------------------------------------------------------------
# 2. Define state structure for LangGraph
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


# ---------------------------------------------------------------------
# 3. Sample documents + Vector store setup
# ---------------------------------------------------------------------
sample_texts = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) combines retrieval with generation, using relevant documents to improve factual accuracy.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They're often used in RAG systems to find related documents semantically.",
    "Agentic systems are AI systems that can make decisions, take actions, and interact with their environment autonomously, often using planning and reasoning."
]

documents = [Document(page_content=text) for text in sample_texts]
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(k=3)


# ---------------------------------------------------------------------
# 4. Define agent node functions
# ---------------------------------------------------------------------
def decide_retrieval(state: AgentState) -> AgentState:
    """Decide if we need retrieval based on the question content."""
    question = state["question"]
    retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]
    needs_retrieval = any(kw in question.lower() for kw in retrieval_keywords)
    print(f"Decide retrieval: {needs_retrieval}")
    return {**state, "needs_retrieval": needs_retrieval}


def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve relevant docs based on the question."""
    question = state["question"]
    docs = retriever.invoke(question)
    return {**state, "documents": docs}


def generate_answer(state: AgentState) -> AgentState:
    """Generate an answer using retrieved context (if any)."""
    question = state["question"]
    docs = state.get("documents", [])

    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are an expert assistant.
Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"Answer the following question clearly and concisely:\n\n{question}"

    response = llm.invoke(prompt)
    answer = response.content.strip()
    return {**state, "answer": answer}


# ---------------------------------------------------------------------
# 5. Define conditional routing
# ---------------------------------------------------------------------
def should_retrieve(state: AgentState) -> str:
    """Decide next step based on retrieval need."""
    return "retrieve" if state["needs_retrieval"] else "generate"


# ---------------------------------------------------------------------
# 6. Build LangGraph workflow
# ---------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", should_retrieve, {
    "retrieve": "retrieve",
    "generate": "generate"
})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()


# ---------------------------------------------------------------------
# 7. Helper function to run the agent
# ---------------------------------------------------------------------
def ask_question(question: str):
    """Run a full agentic RAG pass for a given question."""
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }
    result = app.invoke(initial_state)
    return result


# ---------------------------------------------------------------------
# 8. Demo runs
# ---------------------------------------------------------------------
if __name__ == "__main__":
    q1 = "What is LangGraph?"
    r1 = ask_question(q1)
    print(f"Q: {q1}\nA: {r1['answer']}\nRetrieved docs: {len(r1['documents'])}\n")

    q2 = "When is langchain used?"
    r2 = ask_question(q2)
    print(f"Q: {q2}\nA: {r2['answer']}\nRetrieved docs: {len(r2['documents'])}\n")
