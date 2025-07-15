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