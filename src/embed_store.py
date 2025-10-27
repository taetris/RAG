# embed_store.py
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()
client = OpenAI()

def chunk_text(text, chunk_size=800, overlap=100):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_texts(chunks):
    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding
        embeddings.append(emb)
    return np.array(embeddings, dtype="float32")

def build_index(chunks):
    embeddings = embed_texts(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks
