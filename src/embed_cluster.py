
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
from datasets import load_dataset
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)


# Custom stopwords for BERTopic (sports, Reddit, platform, filler, numbers, etc)
# Only include non-meaningful/filler terms, avoid removing trending/semantic words
BASE_STOPWORDS = set([
    'the', 'and', 'to', 'of', 'in', 'a', 'is', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 'as', 'from', 'that',
    'it', 'are', 'was', 'this', 'will', 'has', 'have', 'but', 'not', 'or', 'its', 'after', 'his', 'he', 'she', 'they',
    'their', 'you', 'we', 'who', 'all', 'about', 'more', 'up', 'out', 'new', 'one', 'over', 'into', 'than', 'just',
    'so', 'can', 'if', 'no', 'how', 'what', 'when', 'which', 'do', 'did', 'been', 'also', 'had', 'would', 'could',
    'should', 'our', 'your', 'them', 'get', 'got', 'like', 'now', 'see', 'us', 'off', 'only', 'back', 'time', 'make',
    'made', 'still', 'very', 'much', 'where', 'why', 'go', 'going', 'may', 'want', 'needs', 'need', 'even', 'most',
    'first', 'last', 'said', 'says', 'year', 'years', 'day', 'days', 'game', 'games', 'season', 'team', 'teams',
    'player', 'players', 'coach', 'coaches', 'match', 'matches', 'win', 'won', 'loss', 'losses', 'play', 'playing',
    'score', 'scored', 'scoring', 'points', 'point', 'home', 'away', 'vs', 'vs.', 'espn', 'reddit', 'category', 'him', 'some'
])
CUSTOM_STOPWORDS = BASE_STOPWORDS.union({
    # General sports filler
    'fan', 'fans', 'ranked', 'rank', 
    'division', 'conference', 'record', 'standings', 'stats', 'stat', 'report', 'rumor'
    , 'healthy', 'practice', 'lineup', 'pts', 'reb', 'ast', 'discussion'
    # Contextual noise
    'thread', 'post', 'title', 'comment', 'op', 'link', 'source',
    'via', 'tweet', 'video', 'article', 'photo', 'highlight', 'clip',
    # Numbers that get picked up
    '2025', '2024', '2023', '10', '20', '30', '40', '50','29', '28', '39', '38', '27', '26', '37', '36', '25', '24', '35', '34', '23', '22', '33', '32', '21', '20', '31', '30', '19', '18', '29', '28', '17', '16', '27', '26', '15', '14', '25', '24', '13', '12', '23', '22', '11', '10', '21', '20', '9', '8', '19', '18', '7', '6', '17', '16', '5', '4', '15', '14', '3', '2', '13', '12', '1', '0', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'
    # Reddit and platform-specific
    'r', 'u', 'subreddit', 'mod', 'nsfw', 'flair', 'discussion thread', 'discussion', 'thread'
    # Sports generic verbs
    'watch', 'watching', 'talk', 'talking', 'looks', 'looking', 'feel', 'feeling',
    'start', 'starting', 'started', 'bench',
    # Outcome-related but vague
    'better', 'worse', 'good', 'bad', 'crazy', 'insane', 'wild', 'amazing', 'great', 'terrible',
    # Random junk
    'etc', 'lol', 'yeah', 'thing', 'stuff', 'weekly discussion', 'weekly', 'footballrelated', 'chat latest'
})


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

def embed_texts(texts, model_name="all-mpnet-base-v2", batch_size=64):
    """
    Generate embeddings for all texts using SentenceTransformers.
    Uses GPU if available.
    models: all-MiniLM-L6-v2, 
    """
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Embedding on device: {device}")
    # Efficient batch encoding
    embeddings = model.encode(texts, show_progress_bar=True, device=device, batch_size=batch_size)
    return embeddings

def cluster_embeddings(embeddings, texts=None, method="bertopic", hdbscan_min_cluster_size=5, bertopic_min_topic_size=5):
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
        if texts is None:
            raise ValueError("BERTopic requires passing `texts` (the list of documents).")
        # Use custom stopwords in vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer_model = TfidfVectorizer(stop_words=list(CUSTOM_STOPWORDS), ngram_range=(1,2), max_features=5000)
        umap_model = UMAP(n_neighbors=10, n_components=10, min_dist=0.0, metric='cosine')
        hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        model = BERTopic(
            min_topic_size=bertopic_min_topic_size,  # try 3, 5, 7, 10
            vectorizer_model=vectorizer_model,
            umap_model=umap_model, 
            hdbscan_model=HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        )
        labels, _ = model.fit_transform(texts, embeddings)
        # Reduce topics to merge highly similar ones and eliminate redundancies
        #model = model.reduce_topics(texts,nr_topics=None)  # Try 10 topics
        #labels = model.topics_
        return labels, model

    else:
        raise ValueError(f"Unknown clustering method: {method}. Choose 'hdbscan' or 'bertopic'.")

def save_results(embeddings, metadata, labels, model=None, method="bertopic"):
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
                clusters[cid] = {"post_ids": []}
            clusters[cid]["post_ids"].append(pid)
    
    
        # BERTopic‑only: enrich clusters with titles & keywords, then save clusters.json
    if method == "bertopic" and model is not None:
        # Extract keyword lists
        keywords_map = {
            cid: [word for word, _ in model.get_topic(cid)]
            for cid in clusters
            if model.get_topic(cid)
        }
        # Extract human‑readable titles
        import pandas as pd
        info = model.get_topic_info()
        titles_map = {
            int(row.Topic): row.Name
            for row in info.itertuples(index=False)
            if row.Topic != -1
        }
        # Merge into clusters dict
        for cid, data in clusters.items():
            clusters[cid]["title"]    = titles_map.get(cid, "")
            clusters[cid]["keywords"] = keywords_map.get(cid, [])

    # Save enriched clusters.json
    with open(os.path.join(DATA_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

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
    labels, model = cluster_embeddings(embeddings, texts=texts, method=method, **kwargs)
    save_results(embeddings, metadata, labels, model=model, method=method)
    print(f"Pipeline complete with method={method}. Produced {len(set(labels))} clusters.")

if __name__ == "__main__":
    # Default: BERTopic clustering
    run_pipeline(method="bertopic")