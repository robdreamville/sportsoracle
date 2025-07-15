import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def load_data(path="data/raw_combined.json"):
    """
    Load combined Reddit + ESPN data, construct a rich 'text' field for embedding.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item["source"] == "reddit":
            # Compose text: title + selftext if available
            base_text = item.get("title", "") or ""
            selftext = item.get("selftext", "") or ""
            item["text_for_embedding"] = f"{base_text}\n\n{selftext}".strip()

        elif item["source"] == "espn":
            # Compose text: title + summary + text + category
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            category = item.get("category", "") or ""
            # Combine all meaningful text fields
            combined_text = f"{title}\n\n{summary}\n\n{text}\n\nCategory: {category}".strip()
            item["text_for_embedding"] = combined_text

        else:
            # Fallback: just use existing text field if any
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
    os.makedirs("data", exist_ok=True)
    np.save("data/embeddings.npy", embeddings)
    with open("data/metadata.jsonl", "w", encoding="utf-8") as f:
        for item in metadata:
            # Remove large text_for_embedding to keep metadata slim
            item.pop("text_for_embedding", None)
            f.write(json.dumps(item) + "\n")
    clusters = [{"id": m["id"], "cluster": int(label)} for m, label in zip(metadata, labels)]
    with open("data/clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)
    print(f"Saved embeddings ({len(embeddings)}), metadata, and clusters.")

def run_pipeline():
    metadata, texts = load_data()
    embeddings = embed_texts(texts)
    labels = cluster_embeddings(embeddings)
    save_results(embeddings, metadata, labels)

if __name__ == "__main__":
    run_pipeline()