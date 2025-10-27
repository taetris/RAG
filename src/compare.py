# compare.py
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

def find_similarities(company_chunks, company_index, regulation_chunks, top_k=2):
    results = []
    for i, reg_chunk in enumerate(regulation_chunks):
        reg_emb = client.embeddings.create(
            input=reg_chunk,
            model="text-embedding-3-small"
        ).data[0].embedding
        _, idx = company_index.search(np.array([reg_emb], dtype="float32"), top_k)
        matches = [company_chunks[j] for j in idx[0]]
        results.append((reg_chunk, matches))
    return results

def interpret_differences(similarity_pairs):
    interpretations = []
    for reg_chunk, company_matches in similarity_pairs:
        prompt = f"""
You are a compliance analyst.

Regulation clause:
{reg_chunk}

Company policy sections:
{company_matches}

Compare them and explain:
- If the company policy aligns or misses anything important
- What potential compliance risk exists
- Suggested fix
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        interpretations.append(resp.choices[0].message.content.strip())
    return interpretations
