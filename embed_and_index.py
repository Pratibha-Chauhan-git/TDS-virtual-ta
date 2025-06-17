import os
import json
import glob
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

def clean_text(text):
    return " ".join(text.strip().split())

def normalize(v):
    return v / np.linalg.norm(v)

# === Load Discourse JSON Data ===
discourse_file = "discourse_posts.json"
if os.path.exists(discourse_file):
    with open(discourse_file, "r", encoding="utf-8") as f:
        discourse_posts = json.load(f)
else:
    discourse_posts = []

# === Load Markdown Course Content ===
markdown_dir = "tds-virtual-ta/course_content"
markdown_files = glob.glob(f"{markdown_dir}/**/*.md", recursive=True)
markdown_data = []
for path in markdown_files:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        markdown_data.append({"path": path, "content": content})

print(f"Loaded {len(discourse_posts)} discourse posts and {len(markdown_data)} markdown files")

# === Group Discourse posts by topic_id ===
topics = {}
for post in discourse_posts:
    topic_id = post["topic_id"]
    if topic_id not in topics:
        topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
    topics[topic_id]["posts"].append(post)

for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

# === Initialize embedding model ===
print("Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Build discourse subthreads ===
def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

embedding_data = []
embeddings = []

print("Encoding discourse subthreads...")
for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    title = topic_data["topic_title"]
    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]

    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)
        combined_text = f"Topic title: {title}\n\n" + "\n\n---\n\n".join(
            clean_text(p["content"]) for p in subthread_posts
        )
        emb = model.encode(combined_text, convert_to_numpy=True)
        emb = normalize(emb)
        embedding_data.append({
            "source": "discourse",
            "topic_id": topic_id,
            "topic_title": title,
            "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}",
            "text": combined_text
        })
        embeddings.append(emb)

print("Encoding markdown documents...")
for md in tqdm(markdown_data):
    emb = model.encode(md["content"], convert_to_numpy=True)
    emb = normalize(emb)
    title = os.path.basename(md["path"]).replace(".md", "")
    url = f"https://tds.s-anand.net/#/{title}"  # Customize if needed
    embedding_data.append({
        "source": "markdown",
        "topic_title": title,
        "url": url,
        "text": md["content"]
    })
    embeddings.append(emb)

print("Building FAISS index...")
embeddings = np.vstack(embeddings).astype("float32")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"Indexed {len(embedding_data)} documents.")

faiss.write_index(index, "vector.index")
with open("embedding_data.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, indent=2)

print("✅ Saved FAISS index and embedding metadata.")
