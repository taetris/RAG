# document.py
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------
# 1. Load your PDF
# -------------------------------
pdf_path = "data/pdf/test.pdf"  # replace with your actual PDF filename
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()  # list of Document objects
print(f"Loaded {len(docs)} pages from PDF.")

# -------------------------------
# 2. Load embedding model
# -------------------------------
# CPU-safe, minimal
model_name = "all-MiniLM-L6-v2"
print(f"Loading embedding model '{model_name}' (first run may download the model)...")
model = SentenceTransformer(model_name)
print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

# -------------------------------
# 3. Embed documents
# -------------------------------
embeddings = []
for i, doc in enumerate(docs, start=1):
    # convert_to_tensor=False is CPU-safe and faster for small number of docs
    embedding = model.encode(doc.page_content, convert_to_tensor=False)
    embeddings.append(embedding)
    print(f"Page {i} embedding shape: {embedding.shape}")

print(f"Generated embeddings for {len(embeddings)} pages.")
