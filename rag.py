import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
# -------------------------------
# 1. Load API key
# -------------------------------

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY") 

# -------------------------------
# 2. Load PDF and chunk
# -------------------------------
pdf_path = "data/pdf/test.pdf"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

chunk_size = 500
chunked_docs = []
for doc in docs:
    text = doc.page_content
    for i in range(0, len(text), chunk_size):
        chunked_docs.append(Document(page_content=text[i:i+chunk_size], metadata=doc.metadata))

# -------------------------------
# 3. Create embeddings
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in chunked_docs]
embeddings = embedding_model.encode(texts, convert_to_tensor=False)
embedding_dim = embeddings[0].shape[0]

# -------------------------------
# 4. Build FAISS vector store
# -------------------------------
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))
index_to_doc = {i: chunked_docs[i] for i in range(len(chunked_docs))}

# -------------------------------
# 5. Retrieval
# -------------------------------
def retrieve(query, top_k=3):
    query_emb = embedding_model.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_emb), top_k)
    return [index_to_doc[i].page_content for i in I[0]]

# -------------------------------
# 6. Generate answer using ChatGPT
# -------------------------------
def generate_answer(query):
    context_chunks = retrieve(query)
    context = "\n".join(context_chunks)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0
    )
    return response.choices[0].message.content

# -------------------------------
# 7. Interactive loop
# -------------------------------
if __name__ == "__main__":
    while True:
        q = input("Enter your question (or 'exit'): ")
        if q.lower() == "exit":
            break
        ans = generate_answer(q)
        print("\n--- Answer ---\n")
        print(ans)
        print("\n--------------\n")
