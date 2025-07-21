# =========================
# SportsOracle: FAISS Indexing & Semantic Search
# =========================
# This script builds a FAISS vector index for fast semantic search over embedded sports data.
# =========================

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import load_config
config = load_config()

# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
def get_project_root():
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def build_faiss_index(
    category="nba",  # or 'soccer'
    model_name=None
):
    """
    Build a FAISS index from precomputed embeddings and save index + id→metadata mapping.
    Now supports category-specific files.
    """
    if model_name is None:
        model_name = config["embedding_model"]
    embeddings_path = os.path.join(DATA_DIR, f"embeddings_{category}.npy")
    metadata_path = os.path.join(DATA_DIR, f"metadata_{category}.jsonl")
    index_path = os.path.join(DATA_DIR, f"faiss_{category}.index")
    mapping_path = os.path.join(DATA_DIR, f"index_mapping_{category}.json")
    # Load embeddings from disk
    embeddings = np.load(embeddings_path)
    # Build FAISS index (L2 distance)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    # Build id -> metadata mapping
    mapping = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            meta = json.loads(line)
            mapping[i] = meta  # index position to metadata
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"FAISS index and mapping saved to {index_path}, {mapping_path}")

def search(query, top_k=5, model_name=None, category="nba"):
    """
    Search the FAISS index for the top_k most similar items to the input query in the given category.
    Returns a list of dicts with metadata and distance.
    """
    if model_name is None:
        model_name = config["embedding_model"]
    index_path = os.path.join(DATA_DIR, f"faiss_{category}.index")
    mapping_path = os.path.join(DATA_DIR, f"index_mapping_{category}.json")
    # Load model and index
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Embed query
    query_emb = model.encode([query])
    D, I = index.search(query_emb.astype(np.float32), top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        meta = mapping.get(str(idx)) or mapping.get(idx)
        results.append({"metadata": meta, "distance": float(dist)})
    return results

# Allow this script to be run standalone for testing
if __name__ == "__main__":
    for cat in config["categories"]:
        build_faiss_index(category=cat)
