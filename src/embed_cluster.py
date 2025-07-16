
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
from bertopic import BERTopic
from hdbscan import HDBSCAN

# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
def get_project_root():
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_data(path=os.path.join(DATA_DIR, "raw_combined_en.jsonl")):
    """
    Load language-normalized Reddit + ESPN data from JSONL using Hugging Face Datasets.
    Only uses *_en fields for embedding.
    Returns:
        metadata: list of dicts (original data, deduped)
        texts: list of str (text for embedding, always English)
    """
    ds = load_dataset("json", data_files=path, split="train")

    # Filter out items with missing/null/empty IDs
    ds = ds.filter(lambda x: x.get("id") and str(x["id"]).strip().lower() != "none")

    # Deduplicate by ID (keep first occurrence)
    def dedup_by_id(batch):
        seen = set()
        keep = []
        for i, id_ in enumerate(batch["id"]):
            if id_ not in seen:
                seen.add(id_)
                keep.append(True)
            else:
                keep.append(False)
        return {k: [v for v, keep_flag in zip(vals, keep) if keep_flag] for k, vals in batch.items()}
    ds = ds.map(lambda x: x, batched=True).map(dedup_by_id, batched=True)

    # Add text_for_embedding field (always English)
    def build_text_for_embedding(example):
        if example["source"] == "reddit":
            base_text = example.get("title_en", "") or ""
            selftext = example.get("selftext_en", "") or ""
            text_for_embedding = f"{base_text}\n\n{selftext}".strip()
        elif example["source"] == "espn":
            title = example.get("title_en", "") or ""
            summary = example.get("summary_en", "") or ""
            text = example.get("text_en", "") or ""
            text_for_embedding = f"{title}\n\n{summary}\n\n{text}".strip()
        else:
            text_for_embedding = example.get("text_en", "")
        example["text_for_embedding"] = text_for_embedding
        return example
    ds = ds.map(build_text_for_embedding)

    # Convert to list of dicts and texts
    metadata = ds.to_list()
    texts = [item["text_for_embedding"] for item in metadata]
    return metadata, texts

def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    """
    Generate embeddings for all texts using SentenceTransformers.
    Uses GPU if available.
    """
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Embedding on device: {device}")
    # Efficient batch encoding
    embeddings = model.encode(texts, show_progress_bar=True, device=device, batch_size=batch_size)
    return embeddings

def cluster_embeddings(embeddings, method="bertopic", hdbscan_min_cluster_size=5, bertopic_min_topic_size=10):
    """
    Cluster embeddings using HDBSCAN or BERTopic.

    Args:
        embeddings: np.ndarray of shape (n_samples, n_features)
        method: 'hdbscan' or 'bertopic'
        hdbscan_min_cluster_size: HDBSCAN minimum cluster size
        bertopic_min_topic_size: BERTopic minimum topic size

    Returns:
        labels: array of cluster/topic labels for each embedding
        model: fitted clustering or topic model
    """
    if method == "hdbscan":
        # HDBSCAN infers clusters and labels noise as -1
        model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        labels = model.fit_predict(embeddings)
        return labels, model

    elif method == "bertopic":
        # BERTopic handles embedding clustering and topic extraction
        model = BERTopic(min_topic_size=bertopic_min_topic_size)
        labels, _ = model.fit_transform(embeddings)
        return labels, model

    else:
        raise ValueError(f"Unknown clustering method: {method}. Choose 'hdbscan' or 'bertopic'.")

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

     # If BERTopic, extract and save topic keywords
    if method == "bertopic" and model is not None and isinstance(model, BERTopic):
        keywords_map = {}
        for cid in clusters:
            topic = model.get_topic(cid)
            # model.get_topic(-1) returns outliers; skip if empty
            if topic:
                keywords_map[cid] = [word for word, _ in topic]
        with open(os.path.join(DATA_DIR, "cluster_keywords.json"), "w", encoding="utf-8") as f:
            json.dump(keywords_map, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings, labels, metadata, clusters{' and keywords' if method=='bertopic' else ''}.")


def run_pipeline(method="bertopic", **kwargs):
    """
    Full pipeline: load data, embed, cluster, and save results.
    Args:
        method: 'hdbscan' or 'bertopic'
        kwargs: passed to cluster_embeddings
    """
    metadata, texts = load_data()
    embeddings = embed_texts(texts)
    labels, model = cluster_embeddings(embeddings, method=method, **kwargs)
    save_results(embeddings, metadata, labels, model=model, method=method)
    print(f"Pipeline complete with method={method}. Produced {len(set(labels))} clusters.")

if __name__ == "__main__":
    # Default: BERTopic clustering
    run_pipeline(method="bertopic")