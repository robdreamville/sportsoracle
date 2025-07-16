"""
language_normalizer.py
Detects language and translates all text fields in raw posts to English, outputting a cleaned JSONL file ready for embedding and clustering.
"""
import os
import json
from datasets import load_dataset
from langdetect import detect
from transformers import pipeline

PROJECT_ROOT = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

lang2translator = {
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en"
}
_translator_cache = {}
def translate_to_en(text, lang):
    if lang == "en" or not text.strip():
        return text
    if lang not in lang2translator:
        return text
    model_name = lang2translator[lang]
    if model_name not in _translator_cache:
        _translator_cache[model_name] = pipeline("translation", model=model_name)
    try:
        return _translator_cache[model_name](text, max_length=2000)[0]["translation_text"]
    except Exception:
        return text

def detect_and_translate_post(post):
    text = post.get("title", "") + " " + post.get("selftext", "") + " " + post.get("summary", "") + " " + post.get("text", "")
    try:
        lang = detect(text[:400]) if text.strip() else "en"
    except Exception:
        lang = "en"
    post["detected_language"] = lang
    # Translate all text fields to English if needed
    for field in ["title", "selftext", "summary", "text"]:
        if field in post and isinstance(post[field], str):
            post[field] = translate_to_en(post[field], lang)
        else:
            post[field] = ""
    return post

def normalize_language(
    input_path=os.path.join(DATA_DIR, "raw_combined.jsonl"),
    output_path=os.path.join(DATA_DIR, "raw_combined_en.jsonl")
):
    """
    Reads raw_combined.jsonl, detects language, translates to English, and writes cleaned JSONL for embedding.
    """
    ds = load_dataset("json", data_files=input_path, split="train")
    ds = ds.map(detect_and_translate_post)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote normalized English posts to {output_path}")

if __name__ == "__main__":
    normalize_language()
