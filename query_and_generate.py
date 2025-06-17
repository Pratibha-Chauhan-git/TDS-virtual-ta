import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === Load FAISS index and metadata ===
embedding_data = json.load(open("embedding_data.json", "r", encoding="utf-8"))
index = faiss.read_index("vector.index")

# === Load the same model used during indexing ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Query to search ===
query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"

query_emb = model.encode(query, convert_to_numpy=True)
query_emb = query_emb / np.linalg.norm(query_emb)
query_emb = query_emb.astype("float32")

# === Perform search ===
D, I = index.search(np.array([query_emb]), 3)

print("\nTop matching subthreads:\n")
for score, idx in zip(D[0], I[0]):
    match = embedding_data[idx]
    print(f"🔍 Score: {score:.4f}")
    print(f"📌 Topic: {match.get('topic_title', 'N/A')}")
    print("📝 Snippet:")
    snippet = match.get("text", "⚠️ No combined_text available")
    print(snippet[:500], "\n---\n")

