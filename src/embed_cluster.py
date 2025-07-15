# =========================
# SportsOracle: Embedding & Clustering Pipeline
# =========================
# This script loads combined sports data, generates embeddings, clusters them,
# and saves the results for downstream analysis (summarization, search, etc).
# =========================

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN

# Directory where all processed data will be stored
DATA_DIR = "/content/sportsoracle/data"

def load_data(path=os.path.join(DATA_DIR, "raw_combined.json")):
    """
    Load combined Reddit + ESPN data and construct a rich 'text_for_embedding' field for each item.
    Handles different formats for Reddit and ESPN sources.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item["source"] == "reddit":
            # Reddit: Use title + selftext
            base_text = item.get("title", "") or ""
            selftext = item.get("selftext", "") or ""
            item["text_for_embedding"] = f"{base_text}\n\n{selftext}".strip()
        elif item["source"] == "espn":
            # ESPN: Use title + summary + text + category
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            category = item.get("category", "") or ""
            item["text_for_embedding"] = f"{title}\n\n{summary}\n\n{text}\n\nCategory: {category}".strip()
        else:
            # Fallback: Use any available text field
            item["text_for_embedding"] = item.get("text", "")

    texts = [item["text_for_embedding"] for item in items]
    return items, texts

def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for all texts using SentenceTransformers.
    Uses GPU if available.
    """
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Using device: {device}")
    embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

def cluster_embeddings(embeddings, method="dbscan", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Cluster embeddings using KMeans or DBSCAN.
    method: 'kmeans' or 'dbscan'
    n_clusters: used only for KMeans
    dbscan_eps, dbscan_min_samples: used only for DBSCAN
    Returns: cluster labels (array)
    """
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        print(f"[INFO] KMeans produced {len(set(labels))} clusters.")
        return labels
    elif method == "dbscan":
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"[INFO] DBSCAN produced {n_clusters} clusters, {n_noise} noise points.")
        return labels
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def save_results(embeddings, metadata, labels):
    """
    Save embeddings, metadata, and cluster assignments to disk for downstream use.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for item in metadata:
            # Remove large text_for_embedding to keep metadata slim
            item.pop("text_for_embedding", None)
            f.write(json.dumps(item) + "\n")
    clusters = [{"id": m["id"], "cluster": int(label)} for m, label in zip(metadata, labels)]
    with open(os.path.join(DATA_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)
    print(f"Saved embeddings ({len(embeddings)}), metadata, and clusters.")

def run_pipeline(method="dbscan", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Run the embedding and clustering pipeline.
    method: 'kmeans' or 'dbscan'
    """
    metadata, texts = load_data()
    embeddings = embed_texts(texts)
    labels = cluster_embeddings(
        embeddings,
        method=method,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples
    )
    save_results(embeddings, metadata, labels)

# Allow this script to be run standalone for testing
if __name__ == "__main__":
    # Default: KMeans. To use DBSCAN, call run_pipeline(method="dbscan", dbscan_eps=0.5, dbscan_min_samples=5)
    run_pipeline()