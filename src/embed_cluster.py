
# =========================
# SportsOracle: Embedding & Clustering Pipeline
# =========================
# Loads combined sports data (Reddit + ESPN), generates embeddings, clusters them,
# and saves the results for downstream analysis (summarization, search, etc).
# Now uses Hugging Face Datasets for efficient, scalable data loading.
# =========================

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from datasets import load_dataset

# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
def get_project_root():
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_data(path=os.path.join(DATA_DIR, "raw_combined.jsonl")):
    """
    Load combined Reddit + ESPN data from JSONL using Hugging Face Datasets.
    Returns:
        metadata: list of dicts (original data, deduped)
        texts: list of str (text for embedding)
    """
    # Load as Hugging Face Dataset for efficient processing
    ds = load_dataset("json", data_files=path, split="train")

    # Deduplicate by ID, filter out missing/null IDs
    seen_ids = set()
    metadata = []
    texts = []
    for item in ds:
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id.lower() == "none":
            continue
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        # Build text for embedding (Reddit: title+selftext, ESPN: title+summary+text)
        if item["source"] == "reddit":
            base_text = item.get("title", "") or ""
            selftext = item.get("selftext", "") or ""
            text_for_embedding = f"{base_text}\n\n{selftext}".strip()
        elif item["source"] == "espn":
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            text_for_embedding = f"{title}\n\n{summary}\n\n{text}".strip()
        else:
            text_for_embedding = item.get("text", "")
        item_out = dict(item)
        item_out["text_for_embedding"] = text_for_embedding
        metadata.append(item_out)
        texts.append(text_for_embedding)
    return metadata, texts

def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for all texts using SentenceTransformers.
    Uses GPU if available.
    """
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Using device: {device}")
    # Efficient batch encoding
    embeddings = model.encode(texts, show_progress_bar=True, device=device, batch_size=64)
    return embeddings

def cluster_embeddings(embeddings, method="kmeans", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Cluster embeddings using KMeans or DBSCAN.
    Args:
        embeddings: np.ndarray or list
        method: 'kmeans' or 'dbscan'
        n_clusters: used only for KMeans
        dbscan_eps, dbscan_min_samples: used only for DBSCAN
    Returns:
        cluster labels (array)
    """
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels
    elif method == "dbscan":
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
        return labels
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def save_results(embeddings, metadata, labels):
    """
    Save embeddings, metadata, and cluster assignments to disk for downstream use.
    Outputs:
        - embeddings.npy: numpy array of embeddings
        - labels.npy: numpy array of cluster labels
        - metadata.jsonl: metadata with cluster assignments (one JSON per line)
        - clusters.json: dict of cluster_id -> list of post IDs
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    # Check alignment
    if not (len(embeddings) == len(metadata) == len(labels)):
        raise ValueError(f"Mismatch: embeddings({len(embeddings)}), metadata({len(metadata)}), labels({len(labels)})")
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(DATA_DIR, "labels.npy"), labels)
    # Save metadata.jsonl and build clusters dict
    clusters = {}
    with open(os.path.join(DATA_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for item, label in zip(metadata, labels):
            item_out = dict(item)
            item_out["cluster"] = int(label)
            f.write(json.dumps(item_out, ensure_ascii=False) + "\n")
            cid = int(label)
            pid = str(item_out.get("id", "")).strip()
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(pid)
    # Save clusters.json for downstream summarization
    with open(os.path.join(DATA_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)



def run_pipeline(method="kmeans", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Run the embedding and clustering pipeline.
    Args:
        method: 'kmeans' or 'dbscan'
        n_clusters: number of clusters for KMeans
        dbscan_eps: epsilon for DBSCAN
        dbscan_min_samples: min samples for DBSCAN
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