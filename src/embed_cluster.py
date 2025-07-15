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

# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
def get_project_root():
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_data(path=os.path.join(DATA_DIR, "raw_combined.json")):
    """
    Load combined Reddit + ESPN data and construct a rich 'text_for_embedding' field for each item.
    Handles different formats for Reddit and ESPN sources.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Deduplicate by ID, filter out missing/null IDs
    seen_ids = set()
    deduped_items = []
    for item in items:
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id.lower() == "none":
            continue  # skip items with missing/null ID
        if item_id in seen_ids:
            continue  # skip duplicates, keep first occurrence
        seen_ids.add(item_id)
        deduped_items.append(item)
    items = deduped_items

    # Subreddit to canonical category mapping and helpers
    SUBREDDIT_TO_CATEGORY = {
        "soccer": "soccer",
        "football": "soccer",
        "premierleague": "soccer",
        "championsleague": "soccer",
        "laliga": "soccer",
        "seriea": "soccer",
        "bundesliga": "soccer",
        "ligue1": "soccer",
        "nba": "nba",
        "nbaoffseason": "nba",
        "nba_draft": "nba",
        "nbacirclejerk": "nba",
    }
    NBA_KEYWORDS = ["nba", "basketball"]
    SOCCER_KEYWORDS = ["soccer", "football", "premier league", "champions league", "laliga", "serie a", "bundesliga", "ligue 1", "futbol"]

    def infer_category(subreddit, title, text):
        sub = (subreddit or "").lower()
        if sub in SUBREDDIT_TO_CATEGORY:
            return SUBREDDIT_TO_CATEGORY[sub]
        for kw in NBA_KEYWORDS:
            if kw in sub:
                return "nba"
        for kw in SOCCER_KEYWORDS:
            if kw in sub:
                return "soccer"
        title_l = (title or "").lower()
        text_l = (text or "").lower()
        for kw in NBA_KEYWORDS:
            if kw in title_l or kw in text_l:
                return "nba"
        for kw in SOCCER_KEYWORDS:
            if kw in title_l or kw in text_l:
                return "soccer"
        return "other"

    for item in items:
        if item["source"] == "reddit":
            base_text = item.get("title", "") or ""
            selftext = item.get("selftext", "") or ""
            item["text_for_embedding"] = f"{base_text}\n\n{selftext}".strip()
            subreddit = item.get("subreddit", "")
            item["category"] = infer_category(subreddit, item.get("title", ""), item.get("text", ""))
        elif item["source"] == "espn":
            title = item.get("title", "") or ""
            summary = item.get("summary", "") or ""
            text = item.get("text", "") or ""
            category = item.get("category", "") or ""
            item["text_for_embedding"] = f"{title}\n\n{summary}\n\n{text}\n\nCategory: {category}".strip()
            item["category"] = category.lower() if category else "espn"
        else:
            item["text_for_embedding"] = item.get("text", "")
            item["category"] = "unknown"

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
    embeddings = model.encode(texts, show_progress_bar=False, device=device)
    return embeddings

def cluster_embeddings(embeddings, method="kmeans", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
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
    """

    os.makedirs(DATA_DIR, exist_ok=True)
    # Check alignment
    if not (len(embeddings) == len(metadata) == len(labels)):
        raise ValueError(f"Mismatch: embeddings({len(embeddings)}), metadata({len(metadata)}), labels({len(labels)})")
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)
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
    np.save(os.path.join(DATA_DIR, "labels.npy"), labels)
    # Save clusters.json for downstream summarization
    with open(os.path.join(DATA_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)


# Run the embedding and clustering pipeline
def run_pipeline(method="kmeans", n_clusters=20, dbscan_eps=0.5, dbscan_min_samples=5):
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