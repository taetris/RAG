# Document Structure

from langchain_core.documents import Document

doc = Document(
    page_content =" thsi is the main text content I am using to create RAG",
    metadata = {
        "source":"example.txt",
        "pages": 1,
        "author": "Tripti Sharma",
        "date_created": "2025-06-01"
    }
)

# print(doc)

# Create a simple txt file

import os

# Create folder relative to the script location
folder_path = os.path.join(os.path.dirname(__file__), "data", "pdf")
os.makedirs(folder_path, exist_ok=True)
print("Folder created at:", folder_path)

### Directory Loader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    folder_path,
    glob="**/*.pdf", ## Pattern to match files  
    loader_cls= PyMuPDFLoader, ##loader class to use
    show_progress=False

)

pdf_documents=dir_loader.load()
# print(pdf_documents)

# Embedding and Vector Store

import numpy as np
import chromadb 
import uuid


