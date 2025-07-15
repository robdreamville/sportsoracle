import os
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict

from transformers import pipeline
from langdetect import detect

STOPWORDS = set([
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

def tokenize(text: str) -> List[str]:
    # Simple word tokenizer, lowercased, strips punctuation
    return re.findall(r"\b\w+\b", text.lower())


def clean_unicode(text: str) -> str:
    """
    Normalize unicode to ASCII, removing smart quotes, dashes, etc.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def extract_text_for_summarization(item):
    """
    Compose a text string for summarization, handling Reddit and ESPN formats, and clean unicode.
    """
    if item.get("source") == "reddit":
        base_text = clean_unicode(item.get("title", "") or "")
        selftext = clean_unicode(item.get("selftext", "") or "")
        text = f"{base_text}\n\n{selftext}".strip()
    elif item.get("source") == "espn":
        title = clean_unicode(item.get("title", "") or "")
        summary = clean_unicode(item.get("summary", "") or "")
        text_field = clean_unicode(item.get("text", "") or "")
        category = clean_unicode(item.get("category", "") or "")
        text = f"{title}\n\n{summary}\n\n{text_field}\n\nCategory: {category}".strip()
    else:
        text = clean_unicode(item.get("text", "") or "")
    return text

def get_translator(lang_code):
    """
    Return a translation pipeline for the given language code to English.
    """
    lang2translator = {
        "es": "Helsinki-NLP/opus-mt-es-en",
        "it": "Helsinki-NLP/opus-mt-it-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "fr": "Helsinki-NLP/opus-mt-fr-en"
    }
    if not hasattr(get_translator, "cache"):
        get_translator.cache = {}
    if lang_code not in lang2translator:
        return None
    model_name = lang2translator[lang_code]
    if model_name not in get_translator.cache:
        get_translator.cache[model_name] = pipeline("translation", model=model_name, device=0)
    return get_translator.cache[model_name]

def summarize_clusters(
    clusters_path="data/clusters.json",
    metadata_path="data/metadata.jsonl",
    out_path="outputs/trends_summary.json",
    top_k_titles=5,
    top_k_keywords=10,
    generate_summary=False,
    top_n_clusters=15,
    summary_models=None
):
    """
    Summarize clusters with keyword extraction, top titles, and optional abstractive summary using Hugging Face transformers.
    Only summarizes top N clusters for NBA and soccer categories.
    Handles Reddit and ESPN data formats for text extraction.
    """
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

    # Helper: is NBA or soccer
    def is_nba_or_soccer(post):
        cat = post.get("category", "").lower()
        return "nba" in cat or "soccer" in cat

    # Compute NBA/soccer post count per cluster
    cluster_cats = {}
    for cid, posts in cluster_posts.items():
        nba_soccer_count = sum(is_nba_or_soccer(p) for p in posts)
        cluster_cats[cid] = nba_soccer_count
    # Select top N clusters by NBA/soccer post count
    top_clusters = sorted(cluster_cats.items(), key=lambda x: x[1], reverse=True)[:top_n_clusters]
    top_cluster_ids = set(cid for cid, count in top_clusters if count > 0)


    # Use only English summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

    summaries = {}
    def safe_summary(summary_sentence):
        if not summary_sentence or not summary_sentence.strip():
            return "[EMPTY]"
        return summary_sentence

    for cid, posts in cluster_posts.items():
        # Use the same text extraction logic as embed_cluster.py
        all_text = " ".join([
            extract_text_for_summarization(post) for post in posts
        ])
        tokens = [t for t in tokenize(all_text) if t not in STOPWORDS and len(t) > 2]
        word_freq = Counter(tokens)
        top_keywords = [w for w, _ in word_freq.most_common(top_k_keywords)]
        # Top titles by upvotes/score if available, else by order
        def score(post):
            return post.get("score", 0) or post.get("upvotes", 0) or 0
        top_titles = sorted(posts, key=score, reverse=True)[:top_k_titles]
        top_titles = [p.get("title", "") for p in top_titles if p.get("title")]
        # Generate summary only for selected clusters
        summary_sentence = ""
        if generate_summary and cid in top_cluster_ids:
            # Concatenate top 5-10 post titles and summaries for input
            top_posts = sorted(posts, key=score, reverse=True)[:10]
            concat_text = " ".join([
                extract_text_for_summarization(p) for p in top_posts
            ])
            concat_text = clean_unicode(concat_text)
            concat_text = concat_text.strip()
            max_chars = 2000
            if len(concat_text) > max_chars:
                concat_text = concat_text[:max_chars]
            if len(concat_text) <= 10:
                summary_sentence = "[No summary – low content]"
            else:
                # Detect language (sample first 400 chars)
                try:
                    lang = detect(concat_text[:400])
                except Exception:
                    lang = "en"
                # Translate to English if needed
                if lang != "en":
                    translator = get_translator(lang)
                    if translator:
                        try:
                            translation = translator(concat_text, max_length=2000)[0]["translation_text"]
                            concat_text = clean_unicode(translation)
                        except Exception:
                            summary_sentence = "[No summary – translation failed]"
                            concat_text = ""
                if not summary_sentence and concat_text:
                    try:
                        summary_out = summarizer(concat_text, max_length=60, min_length=15, do_sample=False)
                        summary_sentence = summary_out[0]["summary_text"].strip()
                        if not summary_sentence:
                            summary_sentence = "[No summary]"
                    except Exception as e:
                        summary_sentence = f"[No summary]"
        if not summary_sentence:
            summary_sentence = "[No summary]"
        summaries[cid] = {
            "cluster_id": cid,
            "total_posts": len(posts),
            "top_titles": top_titles,
            "top_keywords": top_keywords,
            "summary": safe_summary(summary_sentence)
        }
    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_clusters(generate_summary=True)