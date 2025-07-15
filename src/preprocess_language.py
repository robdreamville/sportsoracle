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

def translate_to_en_batch(texts, lang):
    """
    Batch translation for a list of texts in the same language.
    """
    # If English or unsupported language, return as is
    if lang == "en":
        return texts
    if lang not in lang2translator:
        return texts
    model_name = lang2translator[lang]
    if model_name not in _translator_cache:
        _translator_cache[model_name] = pipeline(
            "translation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    # Filter out empty strings to avoid CUDA errors
    nonempty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    nonempty_texts = [texts[i] for i in nonempty_indices]
    translated = list(texts)  # Copy original, will replace non-empty
    if nonempty_texts:
        try:
            results = _translator_cache[model_name](nonempty_texts, max_length=2000, batch_size=16)
            for idx, r in zip(nonempty_indices, results):
                translated[idx] = r["translation_text"]
        except Exception:
            # If translation fails, fall back to original
            pass
    return translated


def detect_and_translate_batch(batch):
    # Each field is a list
    titles = [(t or "") for t in batch.get("title", [])]
    selftexts = [(t or "") for t in batch.get("selftext", [])]
    summaries = [(t or "") for t in batch.get("summary", [])]
    texts = [(t or "") for t in batch.get("text", [])]
    concat_texts = [f"{a} {b} {c} {d}" for a, b, c, d in zip(titles, selftexts, summaries, texts)]
    # Detect language for each post
    langs = []
    unidentified_langs = set()
    for ct in concat_texts:
        try:
            lang = detect(ct[:400]) if ct.strip() else "en"
        except Exception:
            lang = "en"
        langs.append(lang)
        if lang != "en" and lang not in lang2translator:
            unidentified_langs.add(lang)
    if unidentified_langs:
        print(f"[DEBUG] Unidentified languages detected in batch: {sorted(unidentified_langs)}")
    # For each field, translate in batch by language
    def batch_translate_field(field_list, langs):
        # Group by language for efficient batching
        from collections import defaultdict
        out = [None] * len(field_list)
        lang2idx = defaultdict(list)
        for i, l in enumerate(langs):
            lang2idx[l].append(i)
        # For best efficiency, process each language group as a batch
        for l, idxs in lang2idx.items():
            texts_to_translate = [field_list[i] for i in idxs]
            # Only translate if not English and not all empty
            translated = translate_to_en_batch(texts_to_translate, l)
            for i, val in zip(idxs, translated):
                out[i] = val
        return out
    title_en = batch_translate_field(titles, langs)
    selftext_en = batch_translate_field(selftexts, langs)
    summary_en = batch_translate_field(summaries, langs)
    text_en = batch_translate_field(texts, langs)
    # Return new batch dict
    batch_out = dict(batch)
    batch_out["detected_language"] = langs
    batch_out["title_en"] = title_en
    batch_out["selftext_en"] = selftext_en
    batch_out["summary_en"] = summary_en
    batch_out["text_en"] = text_en
    return batch_out


import torch
def preprocess_language(
    in_path=os.path.join(DATA_DIR, "raw_combined.jsonl"),
    out_path=os.path.join(DATA_DIR, "raw_combined_en.jsonl")
):
    """
    Reads a JSONL file, detects language, translates to English in batches, and writes a new JSONL file with *_en fields.
    """
    ds = load_dataset("json", data_files=in_path, split="train")
    # For maximum translation efficiency, sort by language before mapping (optional, but recommended for large datasets)
    # This ensures each batch is mostly a single language, maximizing GPU throughput
    def get_lang(example):
        # Use detected_language if present, else fallback to langdetect
        concat = f"{example.get('title','')} {example.get('selftext','')} {example.get('summary','')} {example.get('text','')}"
        try:
            return detect(concat[:400]) if concat.strip() else "en"
        except Exception:
            return "en"
    # Add a temporary language field for sorting
    ds = ds.map(lambda ex: {**ex, "_tmp_lang": get_lang(ex)})
    ds = ds.sort("_tmp_lang")
    ds = ds.map(detect_and_translate_batch, batched=True, batch_size=32)
    ds = ds.remove_columns(["_tmp_lang"])
    ds.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Wrote language-normalized data to {out_path}")

if __name__ == "__main__":
    preprocess_language()
