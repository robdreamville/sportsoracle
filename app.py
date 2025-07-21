import streamlit as st
import json
import os
from collections import Counter

TRENDS_PATH = "outputs/trends_summary.json"
METADATA_PATH = "data/metadata.jsonl"
TOP_K_TITLES = 5  # Use the same as in summarize_trends.py

# --- Data loading and caching ---
@st.cache_data(show_spinner=False)
def load_trends():
    if not os.path.exists(TRENDS_PATH):
        return []
    with open(TRENDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_metadata_with_clusters():
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- Assign category to each cluster using all posts in that cluster ---
def build_cluster_category_map(metadata):
    cluster_to_categories = {}
    for post in metadata:
        cid = str(post.get("cluster"))
        cat = post.get("category", "").lower()
        if cid and cat in ("nba", "soccer"):
            cluster_to_categories.setdefault(cid, []).append(cat)
    return cluster_to_categories

# Assign category: majority vote, tie or empty defaults to 'nba'
def assign_cluster_category(cid, cluster_to_categories):
    cats = cluster_to_categories.get(str(cid), [])
    if not cats:
        return "nba"
    count = Counter(cats)
    if count["nba"] >= count["soccer"]:
        return "nba"
    else:
        return "soccer"

# --- UI ---
st.set_page_config(page_title="SportsOracle Dashboard", layout="wide")
st.title("SportsOracle Trending Topics Dashboard")

# --- Data ---
trends = load_trends()
metadata = load_metadata_with_clusters()

if not trends:
    st.warning("No trend data found. Please run the pipeline to generate outputs/trends_summary.json.")
    st.stop()

# --- Build cluster_id -> category map ---
cluster_to_categories = build_cluster_category_map(metadata)
for trend in trends:
    trend["category"] = assign_cluster_category(trend["cluster_id"], cluster_to_categories)

# --- Sidebar: Sport Filter ---
sport_options = ["All", "nba", "soccer"]
selected_sport = st.sidebar.selectbox(
    "Filter by sport (NBA, Soccer, or All)", sport_options
)

# --- Sidebar: Refresh Button ---
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# --- Search Box ---
search_query = st.text_input("Search summaries or titles", "")

# --- Filtered Trends ---
def trend_matches(trend):
    # Only show nba or soccer clusters
    if trend["category"] not in ("nba", "soccer"):
        return False
    # Sport filter
    if selected_sport != "All" and trend["category"] != selected_sport:
        return False
    # Search filter
    if search_query.strip():
        q = search_query.lower()
        if q in trend["summary"].lower():
            return True
        for title in trend["top_titles"]:
            if q in title.lower():
                return True
        return False
    return True

filtered_trends = [t for t in trends if trend_matches(t)]

# --- Main: Trend Cards ---
if not filtered_trends:
    st.info("No trends match your filter/search.")
else:
    for trend in filtered_trends:
        with st.container():
            st.subheader(f"Trend {trend['cluster_id']} ({trend['total_posts']} posts)")
            st.markdown(f"**Category:** {trend['category'].capitalize()}")
            st.markdown(f"**Summary:** {trend['summary']}")
            st.markdown(f"**Keywords:** {' | '.join(trend['keywords'])}")
            st.markdown("**Top Titles:**")
            for title in trend["top_titles"]:
                st.markdown(f"- {title}")
            st.markdown("---")
