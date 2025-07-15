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
    Normalize unicode to ASCII, removing smart quotes, dashes, accents, etc.
    """
    if not isinstance(text, str):
        return ""
    # Replace common unicode punctuation with ASCII equivalents
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Normalize and encode to ASCII
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

def translate_text(text, lang, field_name="text", cid=None):
    if lang == "en" or not text.strip():
        return clean_unicode(text)
    translator = get_translator(lang)
    if not translator:
        print(f"[WARN] No translator for lang '{lang}' (cluster {cid}, field {field_name})")
        return clean_unicode(text)
    try:
        translation = translator(text, max_length=2000)[0]["translation_text"]
        translation = clean_unicode(translation)
        return translation
    except Exception as e:
        print(f"[ERROR] Translation failed for lang '{lang}' (cluster {cid}, field {field_name}): {e}")
        return clean_unicode(text)

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
        top_titles_raw = [p.get("title", "") for p in top_titles if p.get("title")]
        # Detect language for this cluster (use concat of all titles for robustness)
        all_titles_text = " ".join([p.get("title", "") for p in posts if p.get("title")])
        try:
            lang_titles = detect(all_titles_text[:400]) if all_titles_text.strip() else "en"
        except Exception:
            lang_titles = "en"
        # Translate top_titles
        top_titles_translated = [translate_text(t, lang_titles, field_name="title", cid=cid) for t in top_titles_raw]
        # Translate top_keywords (join as sentence for batch translation, then split)
        keywords_text = " ".join(top_keywords)
        try:
            lang_keywords = detect(keywords_text[:400]) if keywords_text.strip() else "en"
        except Exception:
            lang_keywords = "en"
        if lang_keywords != "en" and keywords_text.strip():
            translated_kw = translate_text(keywords_text, lang_keywords, field_name="keywords", cid=cid)
            top_keywords_translated = [k.strip() for k in translated_kw.split() if k.strip()]
            # If translation fails or returns empty, fallback to cleaned original
            if not top_keywords_translated:
                top_keywords_translated = [clean_unicode(k) for k in top_keywords]
        else:
            top_keywords_translated = [clean_unicode(k) for k in top_keywords]
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
                if lang != "en":
                    print(f"[INFO] Translating summary input for cluster {cid} from {lang} to en")
                    translator = get_translator(lang)
                    if translator:
                        try:
                            translation = translator(concat_text, max_length=2000)[0]["translation_text"]
                            concat_text = clean_unicode(translation)
                        except Exception as e:
                            print(f"[ERROR] Translation failed for summary in cluster {cid}: {e}")
                            summary_sentence = "[No summary – translation failed]"
                            concat_text = ""
                    else:
                        print(f"[WARN] No translator for summary in cluster {cid} (lang {lang})")
                # Clean unicode again after translation
                concat_text = clean_unicode(concat_text)
                if not summary_sentence and concat_text:
                    try:
                        summary_out = summarizer(concat_text, max_length=60, min_length=15, do_sample=False)
                        summary_sentence = summary_out[0]["summary_text"].strip()
                        summary_sentence = clean_unicode(summary_sentence)
                        if not summary_sentence:
                            summary_sentence = "[No summary]"
                    except Exception as e:
                        print(f"[ERROR] Summarization failed for cluster {cid}: {e}")
                        summary_sentence = f"[No summary]"
        if not summary_sentence:
            summary_sentence = "[No summary]"
        summaries[cid] = {
            "cluster_id": cid,
            "total_posts": len(posts),
            "top_titles": top_titles_translated,
            "top_keywords": top_keywords_translated,
            "summary": safe_summary(summary_sentence)
        }
    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_clusters(generate_summary=True)