# =========================
# SportsOracle: Main Pipeline Entrypoint
# =========================
# This script runs the full SportsOracle data pipeline:
#   1. Scrapes Reddit and ESPN for sports news/posts
#   2. Embeds and clusters the data for downstream analysis
#   3. Summarizes trends and outputs trends_summary.json
#
# To run the full pipeline, simply execute this file.
# =========================

from run_scrape import run_scrape
from src.embed_cluster import run_pipeline
from src.summarize_trends import summarize_trends


def main():
    """
    Run the full SportsOracle pipeline:
      - Scrape Reddit and ESPN
      - Embed and cluster posts (BERTopic/UMAP)
      - Summarize clusters and output trends_summary.json
    """
    print("\n🚀 Running full SportsOracle pipeline...\n")

    # Step 1: Scrape Reddit and ESPN data
    print("[1/3] Scraping Reddit and ESPN data...")
    run_scrape()

    # Step 2: Embed and cluster posts
    print("[2/3] Embedding and clustering posts...")
    run_pipeline()  # Uses BERTopic/UMAP pipeline

    # Step 3: Summarize clusters and output trends_summary.json
    print("[3/3] Summarizing clusters and generating trends_summary.json...")
    summarize_trends()

    print("\n✅ SportsOracle pipeline complete. trends_summary.json is ready for your dashboard.\n")

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
