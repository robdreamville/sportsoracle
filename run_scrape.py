# run_scrape.py
import os
import json

from src.reddit_scraper import scrape_reddit_posts
from src.espn_rss_scraper import scrape_espn_rss

DATA_DIR = "/content/sportsoracle/data"

if __name__ == "__main__":
    # 1️⃣ Your subreddits
    subs = [
        # Soccer
        "soccer", "football", "PremierLeague", "ChampionsLeague",
        "LaLiga", "SerieA", "Bundesliga", "Ligue1",
        # NBA
        "nba", "NBAOffseason", "nba_draft", "nbacirclejerk",
    ]

    # 2️⃣ Scrape Reddit
    print("🔴 Scraping Reddit posts…")
    reddit_posts = scrape_reddit_posts(subs, limit=50, data_dir=DATA_DIR)

    # 3️⃣ Scrape ESPN RSS
    print("🔵 Scraping ESPN RSS…")
    espn_items = scrape_espn_rss(data_dir=DATA_DIR)

    # 4️⃣ Combine & save
    all_items = reddit_posts + espn_items
    os.makedirs(DATA_DIR, exist_ok=True)

    # (a) raw_posts.json already written by reddit_scraper
    # (b) raw_espn.json already written by espn_rss_scraper
    combined_path = os.path.join(DATA_DIR, "raw_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"🔗 Combined {len(all_items)} items → {combined_path}")
    print("✅ Scraping complete!")