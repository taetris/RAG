# main.py
from load_docs import load_docs
from embed_store import chunk_text, build_index
from compare import find_similarities, interpret_differences

def main():
    print("Loading documents...")
    company_doc, regulation_doc = load_docs()

    print("Chunking documents...")
    company_chunks = chunk_text(company_doc)
    regulation_chunks = chunk_text(regulation_doc)

    print("Embedding and indexing...")
    company_index, _ = build_index(company_chunks)
    regulation_index, _ = build_index(regulation_chunks)

    print("Comparing policies...")
    pairs = find_similarities(company_chunks, company_index, regulation_chunks, regulation_index)

    print("Interpreting differences...")
    report = interpret_differences(pairs)

    for i, section in enumerate(report, start=1):
        print(f"\n--- Comparison {i} ---\n{section}\n")

if __name__ == "__main__":
    main()
