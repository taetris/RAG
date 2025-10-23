import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# --- PDF Text Extraction ---
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# --- Text Splitting ---
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# --- Embedding Creation ---
def get_embeddings(chunks):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedder.embed_documents(chunks)

# --- Comparison Logic ---
def compare_versions(chunks_v1, chunks_v2):
    vectors_1 = get_embeddings(chunks_v1)
    vectors_2 = get_embeddings(chunks_v2)

    sim_matrix = cosine_similarity(vectors_1, vectors_2)

    results = []
    for i, chunk1 in enumerate(chunks_v1):
        best_match_idx = sim_matrix[i].argmax()
        best_score = sim_matrix[i][best_match_idx]

        if best_score > 0.95:
            status = "Unchanged"
        elif best_score > 0.7:
            status = "Modified"
        else:
            status = "Removed or New"

        results.append({
            "Section ID": i + 1,
            "Status": status,
            "v1 Snippet": chunk1[:120].replace("\n", " "),
            "v2 Snippet": chunks_v2[best_match_idx][:120].replace("\n", " ") if best_match_idx < len(chunks_v2) else "",
            "Similarity": round(float(best_score), 3)
        })

    return results

# --- Save results to CSV ---
def save_csv(results, output_path):
    header = "Section ID,Status,v1 Snippet,v2 Snippet,Similarity\n"
    lines = [f"{r['Section ID']},{r['Status']},\"{r['v1 Snippet']}\",\"{r['v2 Snippet']}\",{r['Similarity']}" for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines))

# --- Main script ---
def main():
    folder = "data"
    file1 = os.path.join(folder, "version1.pdf")
    file2 = os.path.join(folder, "version2.pdf")

    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Place two PDFs named 'version1.pdf' and 'version2.pdf' in the 'data' folder.")
        return

    print("Reading documents...")
    text1 = extract_text(file1)
    text2 = extract_text(file2)

    print("Splitting and comparing sections...")
    chunks_v1 = split_text(text1)
    chunks_v2 = split_text(text2)

    results = compare_versions(chunks_v1, chunks_v2)

    unchanged = sum(1 for r in results if r['Status'] == 'Unchanged')
    modified = sum(1 for r in results if r['Status'] == 'Modified')
    new_removed = sum(1 for r in results if r['Status'] == 'Removed or New')

    print(f"Unchanged: {unchanged}")
    print(f"Modified: {modified}")
    print(f"New or Removed: {new_removed}")

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "comparison_report.csv")
    save_csv(results, output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()
