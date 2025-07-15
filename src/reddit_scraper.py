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
        for post in tqdm(reddit.subreddit(sub).new(limit=limit),
                         desc=f"Scraping r/{sub}"):
            if not post.stickied:
                # Compose a unified text field (title + selftext)
                text = post.title + "\n\n" + (post.selftext or "")
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