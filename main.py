# =========================
# SportsOracle: Main Pipeline Entrypoint
# =========================
# This script runs the full SportsOracle data pipeline:
#   1. Scrapes Reddit and ESPN for sports news/posts
#   2. Embeds and clusters the data for downstream analysis
#   3. (Planned) Summarizes trends, builds FAISS index, and serves a dashboard
#
# To run the full pipeline, simply execute this file.
# =========================

from run_scrape import run_scrape
from src.embed_cluster import run_pipeline

def main():
    """
    Run the full SportsOracle pipeline:
      - Scrape Reddit and ESPN
      - Embed and cluster posts
      - (Planned) Summarize and index trends
    """
    print("\n🚀 Running full SportsOracle pipeline...\n")

    # Step 1: Scrape Reddit and ESPN data
    run_scrape()

    # Step 2: Embed and cluster posts
    run_pipeline()

    # (Planned) Step 3: Summarize clusters, build FAISS index, serve dashboard
    # from src.summarize_trends import summarize_clusters
    # summarize_clusters(generate_summary=True)
    # from src.faiss_indexer import build_faiss_index
    # build_faiss_index()

    print("\n✅ SportsOracle pipeline complete.\n")

if __name__ == "__main__":
    main()

# =========================
# TODOs & Next Steps
# =========================
# 1. Add stopword filtering to cluster analysis (remove words like "of", "the", "in")
# 2. Automate full pipeline in main.py and test end-to-end
# 3. Add error handling and logging to scrapers and embedding steps
# 4. Build simple keyword/summary extractor per cluster
# 5. Plan integration of semantic search (FAISS or similar)
# 6. Choose lightweight dashboard framework (Streamlit recommended)
# 7. Build UI to display outputs/trends_summary.json with sport filters
# 8. Integrate FAISS semantic search for querying trends/posts
# 9. Add UI controls: search bar, time filters, refresh data button
# 10. Offload heavy GPU tasks (embedding, summarization) to offline/scheduled jobs
# 11. Deploy locally or in cloud container/VM for easy access
# 12. Add simple visualizations (keyword frequency, cluster sizes) for clarity
# =========================
