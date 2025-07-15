# validate_data.py
"""
Validation script for SportsOracle data pipeline.
Checks the integrity and schema of scraped Reddit and ESPN data for compatibility with Hugging Face Datasets and downstream processing.
"""
import os
import json
from pathlib import Path

def validate_jsonl(path, required_fields=None, id_field="id"):
    errors = []
    ids = set()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"Line {count}: JSON decode error: {e}")
                continue
            if required_fields:
                for field in required_fields:
                    if field not in obj:
                        errors.append(f"Line {count}: Missing field '{field}'")
            if id_field and obj.get(id_field) in ids:
                errors.append(f"Line {count}: Duplicate id '{obj.get(id_field)}'")
            ids.add(obj.get(id_field))
    print(f"Validated {count} lines in {path}")
    if errors:
        print(f"\n❌ ERRORS in {path}:")
        for err in errors:
            print("  -", err)
    else:
        print(f"✅ {path} passed validation!")

def main():
    data_dir = Path(os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()) / "data"
    reddit_path = data_dir / "raw_reddit.jsonl"
    espn_path = data_dir / "raw_espn.jsonl"
    # Define required fields for each source
    reddit_fields = ["id", "source", "subreddit", "author", "title", "text", "score", "num_comments", "url", "created_utc"]
    espn_fields = ["id", "source", "category", "title", "text", "link", "published"]
    if reddit_path.exists():
        validate_jsonl(reddit_path, reddit_fields)
    else:
        print(f"{reddit_path} not found.")
    if espn_path.exists():
        validate_jsonl(espn_path, espn_fields)
    else:
        print(f"{espn_path} not found.")

if __name__ == "__main__":
    main()
