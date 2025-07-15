import os
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict

STOPWORDS = set([
    'the', 'and', 'to', 'of', 'in', 'a', 'is', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 'as', 'from', 'that',
    'it', 'are', 'was', 'this', 'will', 'has', 'have', 'but', 'not', 'or', 'its', 'after', 'his', 'he', 'she', 'they',
    'their', 'you', 'we', 'who', 'all', 'about', 'more', 'up', 'out', 'new', 'one', 'over', 'into', 'than', 'just',
    'so', 'can', 'if', 'no', 'how', 'what', 'when', 'which', 'do', 'did', 'been', 'also', 'had', 'would', 'could',
    'should', 'our', 'your', 'them', 'get', 'got', 'like', 'now', 'see', 'us', 'off', 'only', 'back', 'time', 'make',
    'made', 'still', 'very', 'much', 'where', 'why', 'go', 'going', 'may', 'want', 'needs', 'need', 'even', 'most',
    'first', 'last', 'said', 'says', 'year', 'years', 'day', 'days', 'game', 'games', 'season', 'team', 'teams',
    'player', 'players', 'coach', 'coaches', 'match', 'matches', 'win', 'won', 'loss', 'losses', 'play', 'playing',
    'score', 'scored', 'scoring', 'points', 'point', 'home', 'away', 'vs', 'vs.', 'espn', 'reddit', 'category',
])

def tokenize(text: str) -> List[str]:
    # Simple word tokenizer, lowercased, strips punctuation
    return re.findall(r"\b\w+\b", text.lower())

def summarize_clusters(
    clusters_path="data/clusters.json",
    metadata_path="data/metadata.jsonl",
    out_path="outputs/trends_summary.json",
    top_k_titles=5,
    top_k_keywords=10,
    generate_summary=False
):
    # Load clusters
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    # Load metadata
    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    # Build id -> metadata mapping
    id2meta = {item["id"]: item for item in metadata}
    # Group posts by cluster
    cluster_posts = defaultdict(list)
    for entry in clusters:
        cid = entry["cluster"]
        pid = entry["id"]
        meta = id2meta.get(pid)
        if meta:
            cluster_posts[cid].append(meta)
    # Summarize each cluster
    summaries = {}
    for cid, posts in cluster_posts.items():
        all_text = " ".join([
            (post.get("title", "") + " " + post.get("summary", "") + " " + post.get("text", "")).strip()
            for post in posts
        ])
        tokens = [t for t in tokenize(all_text) if t not in STOPWORDS and len(t) > 2]
        word_freq = Counter(tokens)
        top_keywords = [w for w, _ in word_freq.most_common(top_k_keywords)]
        # Top titles by upvotes/score if available, else by order
        def score(post):
            return post.get("score", 0) or post.get("upvotes", 0) or 0
        top_titles = sorted(posts, key=score, reverse=True)[:top_k_titles]
        top_titles = [p.get("title", "") for p in top_titles if p.get("title")]
        # Optional: generate 1-sentence summary (placeholder)
        summary_sentence = ""
        if generate_summary:
            summary_sentence = f"Cluster {cid}: Top keywords: {', '.join(top_keywords[:5])}."
        summaries[cid] = {
            "cluster_id": cid,
            "total_posts": len(posts),
            "top_titles": top_titles,
            "top_keywords": top_keywords,
            "summary": summary_sentence
        }
    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_clusters()
