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
from src.config import load_config
from transformers import AutoTokenizer
import datetime
from dateutil import parser as date_parser
config = load_config()

# Paths
PROJECT_ROOT = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

CLUSTERS_PATH = os.path.join(DATA_DIR, "clusters.json")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.jsonl")
OUT_PATH = os.path.join(OUTPUTS_DIR, "trends_summary.json")

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

def get_safe_max_length(tokenizer, model_name):
    # Known model hard limits
    if 'bart' in model_name.lower():
        return 1024
    if 'pegasus' in model_name.lower():
        return 512
    # Otherwise, use tokenizer.model_max_length, but cap at 4096 for safety
    return min(getattr(tokenizer, 'model_max_length', 1024), 4096)

# Main summarization pipeline
def summarize_trends(
    clusters_path=None,  # Not used, we process per-category
    metadata_path=None,  # Not used, we process per-category
    out_path=OUT_PATH,
    top_k_titles=None,
    summary_max_length=None,
    summary_min_length=None
):
    # Use config values if not provided
    if top_k_titles is None:
        top_k_titles = config["top_k_titles"]
    if summary_max_length is None:
        summary_max_length = config["summary_max_length"]
    if summary_min_length is None:
        summary_min_length = config["summary_min_length"]
    categories = config["categories"]
    all_cluster_summaries = []
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=config["summary_model"], device=device)
    tokenizer = AutoTokenizer.from_pretrained(config["summary_model"])
    max_input_length = get_safe_max_length(tokenizer, config["summary_model"])
    def truncate_to_max_tokens(text, max_tokens):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens, skip_special_tokens=True)
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
            category = cat
            def score(post):
                return post.get("score", 0) or post.get("num_comments", 0) or 0
            top_posts = sorted(posts, key=score, reverse=True)[:top_k_titles]
            top_titles = [p.get("title_en") or p.get("title") or "" for p in top_posts if (p.get("title_en") or p.get("title"))]
            summary_texts = [get_post_text(p) for p in top_posts if get_post_text(p)]
            summary_input = "\n\n".join(summary_texts)
            # Dynamically truncate to model's safe max input length (in tokens)
            summary_input = truncate_to_max_tokens(summary_input, max_input_length)
            if summary_input.strip():
                try:
                    summary = summarizer(summary_input, max_length=summary_max_length, min_length=summary_min_length, do_sample=False)[0]["summary_text"].strip()
                except Exception as e:
                    summary = f"[Summary failed: {e}]"
            else:
                summary = "[No summary: not enough content]"
            # Collect all post dates as UTC timestamps
            def get_post_timestamp(post):
                if post.get("created_utc"):
                    return float(post["created_utc"])
                elif post.get("published"):
                    try:
                        # Try to parse published as ISO8601/date string
                        dt = date_parser.parse(post["published"])
                        return dt.timestamp()
                    except Exception:
                        return None
                else:
                    return None
            created_utcs = [get_post_timestamp(p) for p in posts]
            created_utcs = [ts for ts in created_utcs if ts]
            if created_utcs:
                start_date = datetime.datetime.utcfromtimestamp(min(created_utcs)).strftime("%Y-%m-%d")
                end_date = datetime.datetime.utcfromtimestamp(max(created_utcs)).strftime("%Y-%m-%d")
            else:
                start_date = end_date = None
            all_cluster_summaries.append({
                "cluster_id": int(cid),
                "total_posts": len(posts),
                "category": category,
                "keywords": cluster.get("keywords", []),
                "summary": summary,
                "top_titles": top_titles,
                "start_date": start_date,
                "end_date": end_date
            })
    all_cluster_summaries.sort(key=lambda x: -x["total_posts"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cluster_summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_trends()