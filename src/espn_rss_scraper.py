import os
import json
from tqdm import tqdm
import feedparser

# 1. Define the ESPN RSS feeds you want to pull.
FEEDS = {
    "NBA":    "https://www.espn.com/espn/rss/nba/news",
    "Soccer": "https://www.espn.com/espn/rss/soccer/news",
}

def scrape_espn_rss():
    """
    Scrape the latest articles from ESPN RSS feeds defined above,
    then write them to data/raw_espn.json and return the list.
    """
    articles = []

    # 2. Loop through each sport feed
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

    # 3. Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # 4. Write out to JSON
    out_path = "data/raw_espn.json"
    with open(out_path, "w") as f:
        json.dump(articles, f, indent=2)

    print(f"✅ Saved {len(articles)} ESPN items to {out_path}")
    return articles
