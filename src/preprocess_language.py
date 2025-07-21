import os
import json
import nltk
import re
from datasets import load_dataset, Dataset
import fasttext
import torch
from transformers import pipeline, AutoTokenizer
from collections import defaultdict
import urllib.request
from tqdm import tqdm
import unicodedata

# Set CUDA debugging environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Download NLTK punkt for sentence tokenization
nltk.download('punkt', quiet=True)

PROJECT_ROOT = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Download fasttext language detection model if not already present
def download_fasttext_model(model_path='src/lid.176.bin'):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Downloading FastText language identification model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully.")
    else:
        print("FastText model already exists.")

# Load fasttext model for language detection
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
    "pt": "Helsinki-NLP/opus-mt-roa-en",  # Portuguese → English
    "no": "Helsinki-NLP/opus-mt-nb-en",  # Norwegian → English
    "ja": "Helsinki-NLP/opus-mt-ja-en",  # Japanese → English
    "ru": "Helsinki-NLP/opus-mt-ru-en",  # Russian → English
    "zh": "Helsinki-NLP/opus-mt-zh-en",  # Chinese → English
}

_translator_cache = {}
_tokenizer_cache = {}

def clean_text(text):
    """Clean text by removing control/non-printable characters and normalizing whitespace, but keep emojis, diacritics, and all printable Unicode characters."""
    if not isinstance(text, str):
        return ""
    # Normalize Unicode (NFKC form)
    text = unicodedata.normalize('NFKC', text)
    # Remove control characters (category starts with 'C', except for common whitespace)
    text = ''.join(
        ch for ch in text
        if (unicodedata.category(ch)[0] != 'C' or ch in ('\n', '\t'))
    )
    # Normalize whitespace (replace all whitespace runs with a single space)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, tokenizer, max_length=460, stride=50):
    """
    Split the text into chunks using the tokenizer's overflow feature.
    
    Args:
        text (str): The input text to be chunked.
        tokenizer: The tokenizer for the translation model.
        max_length (int): The maximum number of tokens per chunk (default is 460).
        stride (int): The number of overlapping tokens between chunks (default is 50).
    
    Returns:
        list: A list of text chunks ready for translation.
    """
    text = clean_text(text)
    if not text:
        return [text]
    
    # Tokenize with overflow to handle long texts
    encoded = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_tensors="pt"
    )
    
    # Decode each chunk back to text
    chunks = [tokenizer.decode(ids, skip_special_tokens=True) for ids in encoded["input_ids"]]
    
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
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"[WARN] Could not load translation model for {lang}: {e}")
            return texts
    
    translator = _translator_cache[model_name]
    tokenizer = _tokenizer_cache[model_name]
    nonempty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    translated = list(texts)
    
    for idx in tqdm(nonempty_indices, desc=f"Translating {lang}"):
        text = clean_text(texts[idx])
        if not text:
            continue
        try:
            # Chunk text using the updated function
            to_translate = chunk_text(text, tokenizer, max_length=460)
            results = translator(to_translate, max_length=512, batch_size=4, truncation=True)
            translated_chunks = [r["translation_text"] for r in results]
            translated[idx] = " ".join(translated_chunks)
        except Exception as e:
            print(f"[WARN] Translation failed for {lang} (text: {text[:50]}...): {e}")
            # Fallback to CPU
            try:
                translator.device = -1
                results = translator(to_translate, max_length=512, batch_size=1, truncation=True)
                translated_chunks = [r["translation_text"] for r in results]
                translated[idx] = " ".join(translated_chunks)
                translator.device = 0 if torch.cuda.is_available() else -1
            except Exception as e2:
                print(f"[WARN] CPU fallback failed for {lang} (text: {text[:50]}...): {e2}")
                translated[idx] = text
    
    return translated

def detect_language(text):
    """Detect language using fasttext."""
    text = clean_text(text)
    if not text:
        return "en"
    prediction = fasttext_model.predict(text.replace('\n', ' ')[:200])[0][0]
    return prediction.replace('__label__', '')

def detect_and_translate_batch(batch):
    """Detect language and translate batch fields to English."""
    titles = [(clean_text(t) or "") for t in batch.get("title", [])]
    selftexts = [(clean_text(t) or "") for t in batch.get("selftext", [])]
    summaries = [(clean_text(t) or "") for t in batch.get("summary", [])]
    texts = [(clean_text(t) or "") for t in batch.get("text", [])]
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
    ds = ds.map(detect_and_translate_batch, batched=True, batch_size=16)
    ds.to_json(out_path, force_ascii=False, lines=True)
    print(f"[INFO] Wrote language-normalized data to {out_path}")

if __name__ == "__main__":
    preprocess_language()