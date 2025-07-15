# =========================
# SportsOracle: Data Scraping Pipeline
# =========================
# This script scrapes Reddit and ESPN for sports news/posts and saves the combined data.
# =========================

import os
import json

from src.reddit_scraper import scrape_reddit_posts
from src.espn_rss_scraper import scrape_espn_rss

# Dynamic project root for cross-platform compatibility (Colab, Kaggle, local)
def get_project_root():
    return os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def run_scrape():
    """
    Scrape Reddit and ESPN for sports news/posts and save combined results.
    """
    # List of subreddits to scrape for sports content
    subs = [
        "soccer", "football", "PremierLeague", "ChampionsLeague",
        "LaLiga", "SerieA", "Bundesliga", "Ligue1",
        "nba", "NBAOffseason", "nba_draft", "nbacirclejerk",
    ]

    print("\n🔴 Scraping Reddit posts…")
    reddit_posts = scrape_reddit_posts(subs, limit=50, data_dir=DATA_DIR)

    print("\n🔵 Scraping ESPN RSS…")
    espn_items = scrape_espn_rss(data_dir=DATA_DIR)

    # Combine all items from both sources
    all_items = reddit_posts + espn_items
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save combined data to disk
    combined_path = os.path.join(DATA_DIR, "raw_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\n🔗 Combined {len(all_items)} items → {combined_path}")
    print("✅ Scraping complete!\n")

# Allow this script to be run standalone for testing
if __name__ == "__main__":
    run_scrape()