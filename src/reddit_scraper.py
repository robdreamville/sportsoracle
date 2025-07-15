import os, json
from dotenv import load_dotenv
from praw import Reddit
from tqdm import tqdm

load_dotenv()

def scrape_reddit_posts(subreddits, limit=200, data_dir="data"):
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

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "raw_posts.json")
    with open(out_path, "w") as f:
        json.dump(posts, f, indent=2)

    print(f"✅ {len(posts)} posts saved to {out_path}")
    return posts