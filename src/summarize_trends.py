# =========================
# SportsOracle: Trend Summarization Pipeline
# =========================
# This script loads clustered sports data, cleans and translates all text to English,
# and generates cluster-level summaries, top keywords, and top titles.
# Handles multilingual input and ensures all output is ASCII-clean and in English.


import os
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict
from transformers import pipeline
from langdetect import detect
from keybert import KeyBERT

# -------------------------
# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
# -------------------------

def get_project_root():
    # Use environment variable if set, else default to cwd
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# -------------------------
# Stopwords for keyword filtering
# -------------------------
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

# -------------------------
# Tokenization utility
# -------------------------
def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words, stripping punctuation.
    """
    return re.findall(r"\b\w+\b", text.lower())

# -------------------------
# Unicode normalization utility
# -------------------------
def clean_unicode(text: str) -> str:
    """
    Normalize unicode to ASCII, removing smart quotes, dashes, accents, etc.
    Handles common European punctuation and diacritics.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove emojis
    text = re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]", "", text)
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Replace common unicode punctuation with ASCII equivalents
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Normalize and encode to ASCII
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# -------------------------
# Compose text for summarization (Reddit/ESPN aware)
# -------------------------
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

# -------------------------
# Translation pipeline cache and utility
# -------------------------
def get_translator(lang_code):
    """
    Return a translation pipeline for the given language code to English.
    Uses Hugging Face Helsinki-NLP models for major European languages.
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
    device = 0 if torch.cuda.is_available() else -1
    if model_name not in get_translator.cache:
        get_translator.cache[model_name] = pipeline("translation", model=model_name, device=device)
    return get_translator.cache[model_name]

# -------------------------
# Summarizer pipeline cache and utility
# -------------------------
def get_summarizer(lang_code):
    """
    Return a summarization pipeline for the given language code.
    Uses BART for English, mBART for other languages.
    """
    if not hasattr(get_summarizer, "cache"):
        get_summarizer.cache = {}
    device = 0 if torch.cuda.is_available() else -1
    if lang_code == "en":
        model_name = "facebook/bart-large-cnn"
        cache_key = (model_name, "en")
        if cache_key not in get_summarizer.cache:
            get_summarizer.cache[cache_key] = pipeline(
                "summarization",
                model=model_name,
                device=device
            )
        return get_summarizer.cache[cache_key]
    else:
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        cache_key = (model_name, lang_code)
        if cache_key not in get_summarizer.cache:
            get_summarizer.cache[cache_key] = pipeline(
                "summarization",
                model=model_name,
                device=device
            )
        return get_summarizer.cache[cache_key]

# -------------------------
# Translate any text to English if needed, with logging
# -------------------------
def translate_text(text, lang, field_name="text", cid=None):
    """
    Translate the given text to English using the appropriate translation model,
    based on the detected or provided language code.
    """
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

# -------------------------
# Main cluster summarization pipeline
# -------------------------
def summarize_clusters(
    clusters_path=None,
    metadata_path=None,
    out_path=None,
    top_k_titles=5,
    top_k_keywords=10,
    generate_summary=False,
    top_n_clusters=15,
    summary_models=None,
    chunk_size=300
):
    """
    Summarize clusters with KeyBERT keyword extraction, top titles, and multilingual summarization.
    """
    # Set default paths if not provided
    if clusters_path is None:
        clusters_path = os.path.join(DATA_DIR, "clusters.json")
    if metadata_path is None:
        metadata_path = os.path.join(DATA_DIR, "metadata.jsonl")
    if out_path is None:
        out_path = os.path.join(OUTPUTS_DIR, "trends_summary.json")

    # Use Hugging Face Datasets for robust, universal loading
    from datasets import load_dataset, Dataset, concatenate_datasets
    # Load clusters
    clusters_ds = load_dataset('json', data_files=clusters_path, split='train')
    # Load metadata
    if metadata_path.endswith('.jsonl'):
        metadata_ds = load_dataset('json', data_files=metadata_path, split='train', field=None)
    else:
        metadata_ds = load_dataset('json', data_files=metadata_path, split='train')
    # Build id -> metadata mapping
    id2meta = {item['id']: item for item in metadata_ds}
    # Group posts by cluster
    cluster_posts = defaultdict(list)
    for entry in clusters_ds:
        cid = entry['cluster']
        pid = entry['id']
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

    summaries = {}
    def safe_summary(summary_sentence):
        if not summary_sentence or not summary_sentence.strip():
            return "[EMPTY]"
        return summary_sentence

    # Initialize KeyBERT with the same embedding model as used for clustering
    # Always use GPU if available
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    chunk_size=250
    kw_model = KeyBERT(model=embed_model)

    for cid, posts in cluster_posts.items():
        # --- Keyword extraction ---
        all_text = " ".join([
            extract_text_for_summarization(post) for post in posts
        ]).strip()
        # Improved chunking for KeyBERT
        top_keywords = []
        if all_text and len(all_text.split()) > 5:
            try:
                text_words = all_text.split()
                chunks = [" ".join(text_words[i:i+chunk_size]) for i in range(0, len(text_words), chunk_size)]
                keyword_scores = Counter()
                for chunk in chunks:
                    keybert_keywords = kw_model.extract_keywords(
                        chunk,
                        keyphrase_ngram_range=(1, 2),
                        stop_words=STOPWORDS,
                        top_n=top_k_keywords
                    )
                    for kw, score in keybert_keywords:
                        if kw:
                            keyword_scores[kw] += score
                # Get top N keywords by aggregated score
                top_keywords = [kw for kw, _ in keyword_scores.most_common(top_k_keywords)]
                if not top_keywords:
                    print(f"[DEBUG] KeyBERT returned empty keyword list for cluster {cid}. Text length: {len(all_text)}. Text: {all_text[:200]}...")
                    raise ValueError("KeyBERT returned empty keyword list")
            except Exception as e:
                print(f"[WARN] KeyBERT failed for cluster {cid}: {str(e)} | Text length: {len(all_text)} | Sample: {all_text[:200]}")
                tokens = [t for t in tokenize(all_text) if t not in STOPWORDS and len(t) > 2]
                word_freq = Counter(tokens)
                top_keywords = [w for w, _ in word_freq.most_common(top_k_keywords)]
        else:
            print(f"[DEBUG] Not enough content for KeyBERT in cluster {cid}. Text length: {len(all_text)}. Text: {all_text[:200]}...")
            tokens = [t for t in tokenize(all_text) if t not in STOPWORDS and len(t) > 2]
            word_freq = Counter(tokens)
            top_keywords = [w for w, _ in word_freq.most_common(top_k_keywords)]
        # --- Top titles (by upvotes/score) ---
        def score(post):
            return post.get("score", 0) or post.get("upvotes", 0) or 0
        top_titles = sorted(posts, key=score, reverse=True)[:top_k_titles]
        top_titles_raw = [p.get("title", "") for p in top_titles if p.get("title")]
        # --- Translate top_titles ---
        all_titles_text = " ".join(top_titles_raw)
        try:
            lang_titles = detect(all_titles_text[:400]) if all_titles_text.strip() else "en"
        except Exception:
            lang_titles = "en"
        top_titles_translated = [translate_text(t, lang_titles, field_name="title", cid=cid) for t in top_titles_raw]
        # --- Enhanced language detection: per-post, then dominant ---
        post_langs = []
        for post in posts:
            try:
                post_text = extract_text_for_summarization(post)
                lang = detect(post_text[:400]) if post_text.strip() else "en"
            except Exception:
                lang = "en"
            post_langs.append(lang)
        # Find dominant language
        from collections import Counter as Ctr
        dominant_lang = Ctr(post_langs).most_common(1)[0][0] if post_langs else "en"
        # --- Translate top_titles ---
        keywords_text = " ".join(top_keywords)
        if dominant_lang != "en" and keywords_text.strip():
            translated_kw = translate_text(keywords_text, dominant_lang, field_name="keywords", cid=cid)
            top_keywords_translated = [k.strip() for k in translated_kw.split() if k.strip()]
            if not top_keywords_translated:
                top_keywords_translated = [clean_unicode(k) for k in top_keywords]
        else:
            top_keywords_translated = [clean_unicode(k) for k in top_keywords]
        # --- Summarization (with translation if needed) ---
        summary_sentence = ""
        if generate_summary and cid in top_cluster_ids:
            top_posts = sorted(posts, key=score, reverse=True)[:10]
            concat_text = " ".join([
                extract_text_for_summarization(p) for p in top_posts
            ])
            concat_text = clean_unicode(concat_text)
            concat_text = concat_text.strip()
            max_chars = 2000
            if len(concat_text) > max_chars:
                concat_text = concat_text[:max_chars]
            if len(concat_text) <= 50:
                print(f"[DEBUG] Skipping summary for cluster {cid}: not enough content (len={len(concat_text)})")
                summary_sentence = "[No summary – low content]"
            else:
                lang = dominant_lang
                summarizer = get_summarizer(lang)
                try:
                    if lang == "en":
                        summary_out = summarizer(concat_text, max_length=60, min_length=5, do_sample=False)
                        summary_sentence = summary_out[0]["summary_text"].strip()
                        summary_sentence = clean_unicode(summary_sentence)
                    else:
                        from transformers import MBart50TokenizerFast
                        model_name = "facebook/mbart-large-50-many-to-many-mmt"
                        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                        inputs = tokenizer(concat_text, return_tensors="pt", truncation=True, max_length=512)
                        summarizer.model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
                        with torch.no_grad():
                            summary_ids = summarizer.model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_length=60,
                                min_length=5,
                                num_beams=4,
                                early_stopping=True
                            )
                        summary_sentence = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0].strip()
                        summary_sentence = clean_unicode(summary_sentence)
                        if not summary_sentence:
                            raise ValueError("mBART summary empty")
                except Exception as e:
                    print(f"[ERROR] Summarization failed for cluster {cid} (lang {lang}): {str(e)} | Input len: {len(concat_text)} | Sample: {concat_text[:200]}")
                    # Fallback: translate to English and summarize with BART
                    try:
                        translated_text = translate_text(concat_text, lang, field_name="summary_fallback", cid=cid)
                        bart_summarizer = get_summarizer("en")
                        summary_out = bart_summarizer(translated_text, max_length=60, min_length=5, do_sample=False)
                        summary_sentence = summary_out[0]["summary_text"].strip()
                        summary_sentence = clean_unicode(summary_sentence)
                    except Exception as e2:
                        print(f"[ERROR] Fallback summarization also failed for cluster {cid}: {str(e2)} | Input len: {len(concat_text)} | Sample: {concat_text[:200]}")
                        summary_sentence = f"[No summary]"
        if not summary_sentence:
            print(f"[DEBUG] Final summary empty for cluster {cid}.")
            summary_sentence = "[No summary]"
        # --- Save cluster summary ---
        summaries[cid] = {
            "cluster_id": cid,
            "total_posts": len(posts),
            "top_titles": top_titles_translated,
            "top_keywords": top_keywords_translated,
            "summary": safe_summary(summary_sentence)
        }
    # --- Write output ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote cluster summaries to {out_path}")

if __name__ == "__main__":
    summarize_clusters(generate_summary=True)