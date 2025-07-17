import os
import json
import nltk
from datasets import load_dataset, Dataset
import fasttext
import torch
from transformers import pipeline
from collections import defaultdict
import urllib.request

# Download NLTK punkt for sentence tokenization
nltk.download('punkt', quiet=True)

PROJECT_ROOT = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Download fasttext language detection model if not already present
def download_fasttext_model(model_path='src/lid.176.bin'):
    if not os.path.exists(model_path):
        print("Downloading FastText language identification model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully.")
    else:
        print("FastText model already exists.")

# Load fasttext model for language detection (adjust path if needed)
download_fasttext_model()
fasttext_model = fasttext.load_model('src/lid.176.bin')

lang2translator = {
    "es": "Helsinki-NLP/opus-mt-es-en",  # Spanish → English
    "it": "Helsinki-NLP/opus-mt-it-en",  # Italian → English
    "de": "Helsinki-NLP/opus-mt-de-en",  # German → English
    "fr": "Helsinki-NLP/opus-mt-fr-en",  # French → English
    "af": "Helsinki-NLP/opus-mt-af-en",  # Afrikaans
    "ca": "Helsinki-NLP/opus-mt-ca-en",  # Catalan
    "da": "Helsinki-NLP/opus-mt-da-en",  # Danish
    "id": "Helsinki-NLP/opus-mt-id-en",  # Indonesian
    "et": "Helsinki-NLP/opus-mt-et-en",  # Estonian
    "sv": "Helsinki-NLP/opus-mt-sv-en",  # Swedish
    "fi": "Helsinki-NLP/opus-mt-fi-en",  # Finnish
    "tl": "Helsinki-NLP/opus-mt-tl-en",  # Tagalog
    "pl": "Helsinki-NLP/opus-mt-pl-en",  # Polish
    "pt": "Helsinki-NLP/opus-mt-roa-en",  # Portuguese → English (Romance languages model)
    "no": "Helsinki-NLP/opus-mt-nb-en",  # Norwegian → English (Norwegian Bokmål)
}

_translator_cache = {}

def chunk_text(text, chunk_size=512):
    """Chunk text into pieces <= chunk_size, splitting on sentence boundaries."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def translate_to_en_batch(texts, lang):
    """Batch translation for a list of texts in the same language."""
    if lang == "en" or lang not in lang2translator:
        return texts
    model_name = lang2translator[lang]
    if model_name not in _translator_cache:
        try:
            _translator_cache[model_name] = pipeline(
                "translation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"[WARN] Could not load translation model for {lang}: {e}")
            return texts
    nonempty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    translated = list(texts)
    for idx in nonempty_indices:
        text = texts[idx]
        to_translate = chunk_text(text) if len(text) > 512 else [text]
        try:
            results = _translator_cache[model_name](to_translate, max_length=512, batch_size=16)
            translated_chunks = [r["translation_text"] for r in results]
            translated[idx] = " ".join(translated_chunks)
        except Exception as e:
            print(f"[WARN] Translation failed for {lang}: {e}")
            translated[idx] = text
    return translated

def detect_language(text):
    """Detect language using fasttext."""
    prediction = fasttext_model.predict(text.replace('\n', ' ')[:400])[0][0]
    return prediction.replace('__label__', '')

def detect_and_translate_batch(batch):
    titles = [(t or "") for t in batch.get("title", [])]
    selftexts = [(t or "") for t in batch.get("selftext", [])]
    summaries = [(t or "") for t in batch.get("summary", [])]
    texts = [(t or "") for t in batch.get("text", [])]
    concat_texts = [f"{a} {b} {c} {d}" for a, b, c, d in zip(titles, selftexts, summaries, texts)]
    langs = [detect_language(ct) if ct.strip() else "en" for ct in concat_texts]
    unidentified_langs = {l for l in langs if l != "en" and l not in lang2translator}
    if unidentified_langs:
        print(f"[DEBUG] Unidentified languages detected in batch: {sorted(unidentified_langs)}")

    def batch_translate_field(field_list, langs):
        out = [None] * len(field_list)
        lang2idx = defaultdict(list)
        for i, l in enumerate(langs):
            lang2idx[l].append(i)
        for l, idxs in lang2idx.items():
            texts_to_translate = [field_list[i] for i in idxs]
            translated = translate_to_en_batch(texts_to_translate, l)
            for i, val in zip(idxs, translated):
                out[i] = val
        return out

    batch_out = dict(batch)
    batch_out["detected_language"] = [l if l in lang2translator or l == "en" else "unsupported" for l in langs]
    batch_out["title_en"] = batch_translate_field(titles, langs)
    batch_out["selftext_en"] = batch_translate_field(selftexts, langs)
    batch_out["summary_en"] = batch_translate_field(summaries, langs)
    batch_out["text_en"] = batch_translate_field(texts, langs)
    return batch_out

def preprocess_language(
    in_path=os.path.join(DATA_DIR, "raw_combined.jsonl"),
    out_path=os.path.join(DATA_DIR, "raw_combined_en.jsonl")
):
    """Process JSONL file to detect languages and translate to English."""
    ds = load_dataset("json", data_files=in_path, split="train")
    ds = ds.map(detect_and_translate_batch, batched=True, batch_size=32)
    ds.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Wrote language-normalized data to {out_path}")

if __name__ == "__main__":
    preprocess_language()