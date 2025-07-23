# =========================
# SportsOracle: Main Pipeline Entrypoint
# =========================
# This script runs the full SportsOracle data pipeline:
#   1. Scrapes Reddit and ESPN for sports news/posts
#   2. Normalizes language (translates to English)
#   3. Embeds and clusters the data for downstream analysis
#   4. Summarizes trends and outputs trends_summary.json
#   5. Builds FAISS indexes for NBA and soccer
#
# To run the full pipeline, simply execute this file.
# =========================

from run_scrape import run_scrape
from src.preprocess_language import preprocess_language
from src.embed_cluster import run_pipeline
from src.summarize_trends import summarize_trends
from src.faiss_indexer import build_faiss_index


def main():
    """
    Run the full SportsOracle pipeline:
      - Scrape Reddit and ESPN
      - Normalize language (translate to English)
      - Embed and cluster posts (BERTopic/UMAP)
      - Summarize clusters and output trends_summary.json
      - Build FAISS indexes for NBA and soccer
    """
    print("\n🚀 Running full SportsOracle pipeline...\n")

    # Step 1: Scrape Reddit and ESPN data
    print("[1/5] Scraping Reddit and ESPN data...")
    run_scrape()

    # Step 2: Normalize language (translate to English)
    print("[2/5] Normalizing language and translating posts to English...")
    preprocess_language()

    # Step 3: Embed and cluster posts
    print("[3/5] Embedding and clustering posts (per category)...")
    run_pipeline()  # Uses BERTopic/UMAP pipeline

    # Step 4: Summarize clusters and output trends_summary.json
    print("[4/5] Summarizing clusters and generating trends_summary.json...")
    summarize_trends()

    # Step 5: Build FAISS indexes for NBA and soccer
    print("[5/5] Building FAISS indexes for NBA and soccer...")
    build_faiss_index(category="nba")
    build_faiss_index(category="soccer")

    print("\n✅ SportsOracle pipeline complete. trends_summary.json and FAISS indexes are ready for your dashboard and search.\n")

if __name__ == "__main__":
    main()

# =========================
# Deployment/Automation Planning
# =========================
# - For daily/regular updates: Run this script in Colab or Kaggle to refresh all data and outputs.
# - To update your Streamlit dashboard: Ensure the outputs/trends_summary.json file is accessible to your dashboard (e.g., sync to a shared folder, S3 bucket, or server).
# - For cloud deployment: Consider running this pipeline on a schedule (e.g., with GitHub Actions, Airflow, or a cron job on a VM) and have your dashboard always read the latest trends_summary.json.
# - For local/Colab workflow: Just run `python main.py` and your dashboard will update with the latest data.
# =========================
#TODO: Integrate FAISS indexer 