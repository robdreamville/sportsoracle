# =========================
# SportsOracle: ESPN RSS Scraper
# =========================
# This script scrapes the latest articles from ESPN RSS feeds for NBA and Soccer.
# =========================

import os
import json
from tqdm import tqdm
import feedparser

# Define the ESPN RSS feeds to pull (NBA and Soccer)
FEEDS = {
    "NBA":    "https://www.espn.com/espn/rss/nba/news",
    "Soccer": "https://www.espn.com/espn/rss/soccer/news",
}

def scrape_espn_rss(data_dir=None):
    """
    Scrape the latest articles from ESPN RSS feeds defined in FEEDS.
    Writes results to data/raw_espn.json and returns the list of articles.
    """
    articles = []

    # Dynamic project root for cross-platform compatibility
    if data_dir is None:
        project_root = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
        data_dir = os.path.join(project_root, "data")

    # Loop through each sport feed
    for category, url in FEEDS.items():
        # Parse the RSS feed
        feed = feedparser.parse(url)

        # Iterate over each entry (article) in the feed
        for entry in tqdm(feed.entries, desc=f"Scraping ESPN {category}"):
            # Build a unified text field (title + summary) for embedding later
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            text    = title + "\n\n" + summary

            # Collect the metadata we care about
            articles.append({
                "id":        entry.get("id") or entry.get("link"),
                "source":    "espn",
                "category":  category,
                "title":     title,
                "summary":   summary,
                "text":      text,
                "link":      entry.get("link", ""),
                "published": entry.get("published", "")
            })

    # Save all articles to disk (legacy JSON)
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "raw_espn.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    # Also write as JSONL for Datasets compatibility
    out_jsonl = os.path.join(data_dir, "raw_espn.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved {len(articles)} ESPN items to {out_path} and {out_jsonl}")
    return articles
