# load_docs.py
import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docs():
    company_path = "data/data_protection.pdf"
    regulation_path = "data/GDPR_2019.pdf"

    company_text = extract_text_from_pdf(company_path)
    regulation_text = extract_text_from_pdf(regulation_path)
    return company_text, regulation_text
