# =========================
# SportsOracle: Reddit Scraper
# =========================
# This script scrapes posts from a list of sports-related subreddits using PRAW.
# =========================

import os
import json
from dotenv import load_dotenv
from praw import Reddit
from tqdm import tqdm

load_dotenv()

def scrape_reddit_posts(subreddits, limit=200, data_dir=None):
    # Subreddit to canonical category mapping and helpers (copied from embed_cluster.py)
    SUBREDDIT_TO_CATEGORY = {
        "soccer": "soccer",
        "football": "soccer",
        "premierleague": "soccer",
        "championsleague": "soccer",
        "laliga": "soccer",
        "seriea": "soccer",
        "bundesliga": "soccer",
        "ligue1": "soccer",
        "nba": "nba",
        "nbaoffseason": "nba",
        "nba_draft": "nba",
        "nbacirclejerk": "nba",
    }
    NBA_KEYWORDS = ["nba", "basketball"]
    SOCCER_KEYWORDS = ["soccer", "football", "premier league", "champions league", "laliga", "serie a", "bundesliga", "ligue 1", "futbol"]

    def infer_category(subreddit, title, text):
        sub = (subreddit or "").lower()
        if sub in SUBREDDIT_TO_CATEGORY:
            return SUBREDDIT_TO_CATEGORY[sub]
        for kw in NBA_KEYWORDS:
            if kw in sub:
                return "nba"
        for kw in SOCCER_KEYWORDS:
            if kw in sub:
                return "soccer"
        title_l = (title or "").lower()
        text_l = (text or "").lower()
        for kw in NBA_KEYWORDS:
            if kw in title_l or kw in text_l:
                return "nba"
        for kw in SOCCER_KEYWORDS:
            if kw in title_l or kw in text_l:
                return "soccer"
        return "other"
    """
    Scrape posts from the given list of subreddits using PRAW and save to disk.
    Returns a list of post dicts.
    """
    # Dynamic project root for cross-platform compatibility
    if data_dir is None:
        project_root = os.environ.get("SPORTSORACLE_ROOT") or os.getcwd()
        data_dir = os.path.join(project_root, "data")
    reddit = Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="sportsoracle"
    )

    posts = []
    for sub in subreddits:
        for post in tqdm(reddit.subreddit(sub).new(limit=limit), desc=f"Scraping r/{sub}"):
            if not post.stickied:
                # Compose a unified text field (title + selftext)
                text = post.title + "\n\n" + (post.selftext or "")
                category = infer_category(sub, post.title, text)
                posts.append({
                    "id": post.id,
                    "source": "reddit",
                    "subreddit": sub,
                    "author": post.author.name if post.author else None,
                    "title": post.title,
                    "selftext": post.selftext,
                    "text": text,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "url": post.url,
                    "created_utc": post.created_utc,
                    "category": category,
                })

    # Save all posts to disk (legacy JSON)
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "raw_posts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    # Also write as JSONL for Datasets compatibility
    out_jsonl = os.path.join(data_dir, "raw_reddit.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for post in posts:
            json.dump(post, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ {len(posts)} posts saved to {out_path} and {out_jsonl}")
    return posts