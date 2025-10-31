import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LegalEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Uses OpenAI embeddings.
        """
        self.model_name = model_name
        # Asymmetric prefixes for improved retrieval performance
        self.doc_prefix = "Represent this legal document section: "
        self.query_prefix = "Find regulations about: "

    def build_hierarchical_context(self, chunk):
        """
        Build hierarchical context string for legal chunks.
        This is crucial for maintaining meaning across nested sections.
        """
        context = ""
        if "parent_sections" in chunk and chunk["parent_sections"]:
            context += "Parent Sections: " + " > ".join(chunk["parent_sections"]) + ". "
        if "section_title" in chunk:
            context += f"Title: {chunk['section_title']}. "
        if "text" in chunk:
            context += f"Text: {chunk['text']}"
        return context.strip()

    def embed_chunks(self, chunks, batch_size=50, show_progress=True):
        """
        Embed all document chunks using OpenAI embeddings.
        Returns a numpy array of shape (n_chunks, embedding_dim)
        """
        all_embeddings = []
        iterator = tqdm(range(0, len(chunks), batch_size), disable=not show_progress)

        for i in iterator:
            batch = chunks[i:i + batch_size]

            # Build prefixed text with hierarchical context
            texts = [
                self.doc_prefix + self.build_hierarchical_context(chunk)
                for chunk in batch
            ]

            # API call
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )

            # Extract embeddings
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a search query using a separate asymmetric prefix.
        """
        prefixed_query = self.query_prefix + query_text
        response = client.embeddings.create(
            model=self.model_name,
            input=[prefixed_query]
        )
        return np.array(response.data[0].embedding)
    

import json

import config
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
if __name__ == "__main__":
    # File paths
    policy_path = Path(config.OUTPUT_FOLDER) / "policy_chunks.json"
    regulation_path = Path(config.OUTPUT_FOLDER) / "regulation_chunks.json"

    embedder = LegalEmbedder()

    # --- POLICY ---
    print("Loading policy chunks...")
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_chunks = json.load(f)
    policy_embeddings = embedder.embed_chunks(policy_chunks)
    print(f"Policy embeddings shape: {policy_embeddings.shape}")

    # --- REGULATION ---
    print("\nLoading regulation chunks...")
    with open(regulation_path, "r", encoding="utf-8") as f:
        regulation_chunks = json.load(f)
    regulation_embeddings = embedder.embed_chunks(regulation_chunks)
    print(f"Regulation embeddings shape: {regulation_embeddings.shape}")

    # --- Combine for visualization ---
    import numpy as np
    combined_embeddings = np.vstack((policy_embeddings, regulation_embeddings))
    labels = (["Policy"] * len(policy_embeddings)) + (["Regulation"] * len(regulation_embeddings))

    # --- PCA projection ---
    print("\nReducing embeddings to 2D for visualization...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined_embeddings)

    plt.figure(figsize=(8, 6))
    for label, color in [("Policy", "blue"), ("Regulation", "red")]:
        mask = [l == label for l in labels]
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=label, alpha=0.7)

    plt.title("Policy vs Regulation Embeddings (PCA projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()

    # --- Optional: Query embedding ---
    query = "data retention under GDPR"
    q_emb = embedder.embed_query(query)
    print(f"\nQuery embedding length: {len(q_emb)}")

    
# import json
# import config

# if __name__ == "__main__":
#     # Load your chunks (already processed)
#     chunks_path = config.OUTPUT_FOLDER / "regulation_chunks.json"

#     with open(chunks_path, "r", encoding="utf-8") as f:
#         chunks = json.load(f)

#     embedder = LegalEmbedder()
#     embeddings = embedder.embed_chunks(chunks)

#     print("Embeddings shape:", embeddings.shape)

#     query = "data retention under GDPR"
#     q_emb = embedder.embed_query(query)
#     print("Query embedding length:", len(q_emb))

#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt

#     # Reduce embeddings to 2D for visualization
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(embeddings)

#     plt.scatter(reduced[:, 0], reduced[:, 1])
#     plt.title("Legal document embeddings (PCA projection)")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.show()
