"""
preprocess_language.py
Detects language and translates all relevant text fields to English for each post in a JSONL file.
Outputs a new JSONL file with *_en fields and detected_language for each post.
"""
import os
import json
from datasets import load_dataset, Dataset
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
    for field in ["title", "selftext", "summary", "text"]:
        if field in post and isinstance(post[field], str):
            post[field + "_en"] = translate_to_en(post[field], lang)
        else:
            post[field + "_en"] = ""
    return post

def preprocess_language(
    in_path=os.path.join(DATA_DIR, "raw_combined.jsonl"),
    out_path=os.path.join(DATA_DIR, "raw_combined_en.jsonl")
):
    """
    Reads a JSONL file, detects language, translates to English, and writes a new JSONL file with *_en fields.
    """
    ds = load_dataset("json", data_files=in_path, split="train")
    ds = ds.map(detect_and_translate_post)
    ds.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Wrote language-normalized data to {out_path}")

if __name__ == "__main__":
    preprocess_language()
