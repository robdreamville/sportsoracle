# SportsOracle 🏀⚽

**TL;DR**: SportsOracle automatically scrapes Reddit & ESPN, translates content to English, and uses GPU‑accelerated embeddings + BERTopic to surface and summarize trending NBA & soccer topics, all wrapped in a Colab/Kaggle‑ready pipeline.

---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robdreamville/sportsoracle/blob/main/notebooks/sportsoracle-pipeline.ipynb)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/robdreamville/sportsoracle/blob/main/notebooks/sportsoracle-pipeline.ipynb)


## 🎯 Why It Matters

Real‑time sports chatter is scattered across subreddits and news feeds—hard to track the biggest conversations. SportsOracle does the heavy lifting:

- ✅ Scrapes your favorite sports subreddits & ESPN RSS  
- ✅ Translates non‑English posts to English  
- ✅ Embeds & Clusters with MiniLM + BERTopic (UMAP + HDBSCAN)  
- ✅ Summarizes each trend with Pegasus/XSum or BART/CNN  
- ✅ Visualizes clusters in an interactive Streamlit dashboard  

> 🧠 Get a single dashboard that tells you what’s buzzing in NBA and soccer.

---

## 🚀 Getting Started (Colab / Kaggle)

1. **Clone this repo:**
   ```bash
   git clone https://github.com/yourusername/SportsOracle.git
   cd SportsOracle
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up Reddit API credentials:

Create an app at https://reddit.com/prefs/apps and add these as environment variables (or in a .env file):

bash
Copy
Edit
export REDDIT_CLIENT_ID=xxx
export REDDIT_CLIENT_SECRET=xxx
export REDDIT_USER_AGENT="script:SportsOracle:v1.0 (by u/yourusername)"
Run full pipeline in a notebook (free GPU):

In Colab or Kaggle, in a notebook cell:

bash
Copy
Edit
# Install & configure
!pip install -r requirements.txt
%env REDDIT_CLIENT_ID=xxx
%env REDDIT_CLIENT_SECRET=xxx
%env REDDIT_USER_AGENT="script:SportsOracle:v1.0 (by u/yourusername)"

# Time the full pipeline
!time python main.py
At the end you’ll see:

pgsql
Copy
Edit
Wrote cluster summaries to /path/to/SportsOracle/outputs/trends_summary.json
FAISS indexes saved to data/faiss_{nba|soccer}.index + index_mapping_*.json
✅ SportsOracle pipeline complete.
Download trends_summary.json:

In Colab: click the file browser ➔ navigate to outputs/ ➔ right‑click trends_summary.json ➔ Download

In Kaggle: use the “Save and Copy Output” feature or the file browser to download from outputs/

Place your downloaded trends_summary.json into your local SportsOracle/outputs/ folder.

💻 Local Dashboard (Streamlit)
After you’ve run the pipeline and downloaded outputs/trends_summary.json:

Ensure outputs/trends_summary.json exists (the app will prompt if missing).

Launch the dashboard:

bash
Copy
Edit
streamlit run app.py
Explore trending topics by sport, click into summaries, and enjoy the UI!

⚙️ Configuration
All pipeline parameters live in config.yaml—no code edits needed:

yaml
Copy
Edit
# Embedding
embedding_model: all-mpnet-base-v2
embedding_batch_size: 64

# BERTopic/Clustering
bertopic_min_topic_size: 10
hdbscan_min_cluster_size: 10
hdbscan_min_samples: 2
hdbscan_metric: euclidean
umap_n_neighbors: 15
umap_n_components: 5
umap_min_dist: 0.1
umap_metric: cosine
vectorizer_max_features: 2000
vectorizer_ngram_range: [1, 3]

# Summarization
summary_model: google/pegasus-xsum       # or facebook/bart-large-cnn
summary_max_length: 80
summary_min_length: 10
top_k_titles: 5

# Categories to process
categories: [nba, soccer]
Tweak these values to experiment with topic granularity and summary style.

🗂️ Project Structure
bash
Copy
Edit
SportsOracle/
├── main.py            # Orchestrates the full pipeline
├── config.yaml        # All parameters (embedding, clustering, summary)
├── requirements.txt
├── src/               # Core modules (scraper, preprocess, embed, cluster, summarize)
├── data/              # Raw & processed data, embeddings, indexes
├── outputs/           # trends_summary.json, visualizations
├── app.py             # Streamlit dashboard
└── README.md
💡 Future Work
Add FAISS‑powered search API

Deploy as a hosted dashboard (optional)

Automate fresh trends (cron jobs, GitHub Actions)

👤 About Me
I’m a data‑obsessed developer who loves bringing AI to real‑world problems.
SportsOracle showcases my skills in NLP pipelines, GPU‑acceleration, and interactive dashboards.
Let’s chat if you want to collaborate or build on top of this!