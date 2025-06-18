import os
import json
import base64
import io
import numpy as np
import faiss
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import pytesseract
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("OPENAI_API_KEY")
if not AIPROXY_TOKEN:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# Load sentence transformer and FAISS index
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
index = faiss.read_index("vector.index")
with open("embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# Final system instructions
SYSTEM_INSTRUCTIONS = """
You are a Teaching Assistant (TA) for the Tools in Data Science course at IIT Madras.

1. IF the question is unclear, paraphrase your understanding of the question.
2. Cite all relevant sections from `tds-content.xml` or `ga*.md`. Begin with: "According to [this reference](https://tds.s-anand.net/#/...), ...". Cite ONLY from the relevant <source>. ALWAYS cite verbatim. Mention ALL material relevant to the question.
3. Search online for additional answers. Share results WITH CITATION LINKS.
4. Think step-by-step. Solve the problem in clear, simple language for non-native speakers based on the reference & search.
5. Follow-up: Ask thoughtful questions to help students explore and learn.
"""

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: Optional[str] = None

# Utility functions
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def call_gpt(messages):
    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": messages,
        "temperature": 0.5
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling GPT: {e}"

@app.post("/api/")
async def answer_question(query: Query):
    query_text = query.question

    # OCR if image provided
    if query.image:
        try:
            image_data = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_data))
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                query_text += f"\n\n[Extracted text from image:]\n{extracted_text.strip()}"
        except Exception as e:
            return {"answer": f"Image processing failed: {e}", "links": []}

    # Embed question and search FAISS
    emb = normalize(model.encode(query_text, convert_to_numpy=True))
    D, I = index.search(np.array([emb]).astype("float32"), k=3)

    top_chunks = []
    links = []
    for idx in I[0]:
        chunk = embedding_data[idx]
        title = chunk.get("topic_title", "No Title")
        source = chunk.get("source", "unknown")

        if source == "discourse":
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{chunk.get('topic_id', 'unknown')}"
        else:
            filename = title.replace(".md", "")
            url = f"https://tds.s-anand.net/#/{filename}"

        links.append({"url": url, "text": title})
        top_chunks.append(f"Source: {source.upper()} | Title: {title} | [Link]({url})\n\n{chunk['text'][:1000]}\n")


    full_prompt = f"""
{SYSTEM_INSTRUCTIONS}

Student's Question:
{query.question}

Reference Material:
{chr(10).join(top_chunks)}

Answer:
"""

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": full_prompt}
    ]
    response = call_gpt(messages)

    return {"answer": response.strip(), "links": links}

# Only for local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
