import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATA_DIR = "/content/sportsoracle/data"

def load_data(path=os.path.join(DATA_DIR, "raw_combined.json")):
    """
    Load combined Reddit + ESPN data, construct a rich 'text' field for embedding.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item["source"] == "reddit":
            base_text = item.get("title", "") or ""
            selftext = item.get("selftext", "") or ""
            item["text_for_embedding"] = f"{base_text}\n\n{selftext}".strip()
        elif item["source"] == "espn":
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            category = item.get("category", "") or ""
            item["text_for_embedding"] = f"{title}\n\n{summary}\n\n{text}\n\nCategory: {category}".strip()
        else:
            item["text_for_embedding"] = item.get("text", "")

    texts = [item["text_for_embedding"] for item in items]
    return items, texts

def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Using device: {device}")
    embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

def cluster_embeddings(embeddings, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def save_results(embeddings, metadata, labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for item in metadata:
            item.pop("text_for_embedding", None)
            f.write(json.dumps(item) + "\n")
    clusters = [{"id": m["id"], "cluster": int(label)} for m, label in zip(metadata, labels)]
    with open(os.path.join(DATA_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)
    print(f"Saved embeddings ({len(embeddings)}), metadata, and clusters.")

def run_pipeline():
    metadata, texts = load_data()
    embeddings = embed_texts(texts)
    labels = cluster_embeddings(embeddings)
    save_results(embeddings, metadata, labels)

if __name__ == "__main__":
    run_pipeline()