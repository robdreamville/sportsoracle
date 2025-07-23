# SportsOracle 🏀⚽

**TL;DR**: A Colab/Kaggle‑ready pipeline that scrapes Reddit & ESPN, translates to English, embeds & clusters with BERTopic, and summarizes trending NBA & soccer topics—then serves them in a Streamlit dashboard.

---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robdreamville/sportsoracle/blob/main/notebooks/sportsoracle-pipeline.ipynb)

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/robdreamville/sportsoracle/blob/main/notebooks/sportsoracle-pipeline.ipynb)

---

## 🎯 Why It Matters

Real‑time sports chatter is scattered across subreddits and news feeds—hard to track the biggest conversations. SportsOracle does the heavy lifting:

- ✅ Scrapes your favorite sports subreddits & ESPN RSS  
- ✅ Translates non‑English posts to English  
- ✅ Embeds with all‑mpnet‑base‑v2 + clusters via BERTopic (UMAP + HDBSCAN)
- ✅ Summarizes each trend with Pegasus/XSum or BART/CNN  
- ✅ Serves an interactive Streamlit dashboard

- 🧠 Get a single dashboard that tells you what’s buzzing in NBA and soccer.

---

## 🚀 Getting Started (Colab / Kaggle)

1. **Open the pipeline notebook**
   - Click the badge above to launch notebooks/sportsoracle-pipeline.ipynb in Colab or Kaggle.

2. **Plug in your Reddit API keys**
    In the first cell, replace the placeholder ENV vars with your own:
    - os.environ["REDDIT_CLIENT_ID"]     = "<YOUR_CLIENT_ID>"
    - os.environ["REDDIT_CLIENT_SECRET"] = "<YOUR_CLIENT_SECRET>"
 
    You can get these by creating an app at https://reddit.com/prefs/apps.

3. **Install & Run**
    In the notebook cells it will:
    - !pip install -r requirements.txt
    - !time python main.py
   
     At the end you’ll see:
    - Wrote cluster summaries to …/outputs/trends_summary.json
    - FAISS indexes saved to data/faiss_{nba|soccer}.index + index_mapping_*.json
    - ✅ SportsOracle pipeline complete.

4. **Download trends_summary.json from the notebook’s outputs/ folder.**

---

## 💻 Local Dashboard (Streamlit Only)
**You don’t need the full pipeline or all dependencies locally—just Streamlit:**

1. **if running the notebook above:** Place your downloaded trends_summary.json into outputs/. 

    **If not:** use the one provided all though it is not current data

2. **Install Streamlit:**
    - pip install streamlit

3. **Run the dashboard:**
    - streamlit run app.py

---

⚙️ Configuration
All parameters are in config.yaml—no code changes required. Example:

**Embedding**
embedding_model: all-mpnet-base-v2
embedding_batch_size: 64

**BERTopic/Clustering**
bertopic_min_topic_size: 10
hdbscan_min_cluster_size: 10
umap_n_neighbors: 15
umap_n_components: 5
vectorizer_ngram_range: [1, 3]

**Summarization**
summary_model: google/pegasus-xsum  # or facebook/bart-large-cnn
summary_max_length: 80
summary_min_length: 10
top_k_titles: 5

**Categories**
categories: [nba, soccer]
Tweak these to dial in cluster granularity and summary style.

---

🗂️ Project Structure

Directory structure:
└── robdreamville-sportsoracle/
    ├── README.md
    ├── app.py                      # Streamlit dashboard (only needs streamlit)
    ├── config.yaml                 # All pipeline settings
    ├── faiss_api.py
    ├── main.py                     # Orchestrates the full pipeline
    ├── requirements.txt            # Full pipeline deps
    ├── run_scrape.py
    ├── sportsoracle_architecture.txt
    ├── validate_data.py
    ├── notebooks/                  # Colab/Kaggle pipeline notebook
    │   └── sportsoracle-pipeline.ipynb
    ├── outputs/
    │   └── trends_summary.json     # trends_summary.json, visualizations
    └── src/                        # Modules: scraper, preprocess, embed, cluster, summarize
        ├── config.py
        ├── embed_cluster.py
        ├── espn_rss_scraper.py
        ├── faiss_indexer.py
        ├── preprocess_language.py
        ├── reddit_scraper.py
        └── summarize_trends.py

---
💡 Future Work
- FAISS‑powered search API

- Hosted dashboard deployment

- Automated updates via cron/GitHub Actions

---
👤 About Me
I’m a data‑obsessed developer who loves applying AI to real‑world problems. SportsOracle highlights end‑to‑end NLP pipelines, GPU acceleration, and interactive dashboards. Let’s build awesome things together!