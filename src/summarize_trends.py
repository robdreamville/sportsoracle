# =========================
# SportsOracle: Trend Summarization Pipeline
# =========================
# This script loads clustered sports data, cleans and translates all text to English,
# and generates cluster-level summaries, top keywords, and top titles.
# Handles multilingual input and ensures all output is ASCII-clean and in English.

import os
import json
from datasets import load_dataset
from transformers import pipeline
import torch
from collections import Counter

# Paths
PROJECT_ROOT = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

CLUSTERS_PATH = os.path.join(DATA_DIR, "clusters.json")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.jsonl")
OUT_PATH = os.path.join(OUTPUTS_DIR, "trends_summary.json")

TOP_K_TITLES = 5
SUMMARY_MAX_LENGTH = 80
SUMMARY_MIN_LENGTH = 10

# Compose text for summarization
def get_post_text(post):
    if post.get("source") == "reddit":
        return f"{post.get('title_en', '')}\n\n{post.get('selftext_en', '')}".strip()
    elif post.get("source") == "espn":
        return f"{post.get('title_en', '')}\n\n{post.get('summary_en', '')}\n\n{post.get('text_en', '')}".strip()
    else:
        return post.get("text_en", "")

# Assign category to a cluster based on majority of post categories
def assign_cluster_category(posts):
    cats = [p.get("category", "").lower() for p in posts if p.get("category", "").lower() in ("nba", "soccer")]
    if not cats:
        return "nba"
    count = Counter(cats)
    if count["nba"] >= count["soccer"]:
        return "nba"
    else:
        return "soccer"

# Main summarization pipeline
def summarize_trends(
    clusters_path=None,  # Not used, we process per-category
    metadata_path=None,  # Not used, we process per-category
    out_path=OUT_PATH,
    top_k_titles=TOP_K_TITLES,
    summary_max_length=SUMMARY_MAX_LENGTH,
    summary_min_length=SUMMARY_MIN_LENGTH
):
    # Process both NBA and soccer clusters/metadata
    categories = ["nba", "soccer"]
    all_cluster_summaries = []
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    for cat in categories:
        clusters_file = os.path.join(DATA_DIR, f"clusters_{cat}.json")
        metadata_file = os.path.join(DATA_DIR, f"metadata_{cat}.jsonl")
        if not (os.path.exists(clusters_file) and os.path.exists(metadata_file)):
            print(f"Missing data for category: {cat}, skipping.")
            continue
        with open(clusters_file, "r", encoding="utf-8") as f:
            clusters = json.load(f)
        metadata = {str(item["id"]): item for item in load_dataset("json", data_files=metadata_file, split="train")}
        for cid, cluster in clusters.items():
            post_ids = cluster.get("post_ids", [])
            posts = [metadata[pid] for pid in post_ids if pid in metadata]
            # Assign category (fixed per file)
            category = cat
            # Top titles by score (or num_comments as fallback)
            def score(post):
                return post.get("score", 0) or post.get("num_comments", 0) or 0
            top_posts = sorted(posts, key=score, reverse=True)[:top_k_titles]
            top_titles = [p.get("title_en") or p.get("title") or "" for p in top_posts if (p.get("title_en") or p.get("title"))]
            # Compose text for summary (use top N posts)
            summary_texts = [get_post_text(p) for p in top_posts if get_post_text(p)]
            summary_input = "\n\n".join(summary_texts)[:2000]  # Truncate to fit model
            # Generate summary
            if summary_input.strip():
                try:
                    summary = summarizer(summary_input, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)[0]["summary_text"].strip()
                except Exception as e:
                    summary = f"[Summary failed: {e}]"
            else:
                summary = "[No summary: not enough content]"
            # Compose output dict
            all_cluster_summaries.append({
                "cluster_id": int(cid),
                "total_posts": len(posts),
                "category": category,
                "keywords": cluster.get("keywords", []),
                "summary": summary,
                "top_titles": top_titles
            })
    # Sort by total_posts descending
    all_cluster_summaries.sort(key=lambda x: -x["total_posts"])
    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cluster_summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_trends()