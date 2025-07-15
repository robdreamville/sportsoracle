# main.py

from run_scrape import run_scrape
from src.embed_cluster import run_pipeline

def main():
    print("🚀 Running full SportsOracle pipeline...")

    run_scrape()
    run_pipeline()

    print("✅ Done.")

if __name__ == "__main__":
    main()

#TODO: 1. Add stopword filtering to cluster analysis (remove words like "of", "the", "in")
#TODO 2. Automate full pipeline in main.py and test end-to-end
#TODO 3. Add error handling and logging to scrapers and embedding steps
#TODO 4. Build simple keyword/summary extractor per cluster
#TODO 5. Plan integration of semantic search (FAISS or similar)