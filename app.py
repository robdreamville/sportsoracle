import streamlit as st
import json
import os
import re

TRENDS_PATH = "outputs/trends_summary.json"

# --- Data loading and caching ---
@st.cache_data(show_spinner=False)
def load_trends():
    if not os.path.exists(TRENDS_PATH):
        return []
    with open(TRENDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Utility: Pick best title for a cluster based on summary ---
def pick_best_title(summary, titles):
    # Lowercase and tokenize, remove punctuation
    def tokenize(text):
        return set(re.findall(r'\w+', text.lower()))
    summary_tokens = tokenize(summary)
    best_title = titles[0] if titles else "(No title)"
    best_score = 0
    for title in titles:
        title_tokens = tokenize(title)
        score = len(summary_tokens & title_tokens)
        if score > best_score:
            best_score = score
            best_title = title
    return best_title

# --- UI ---
st.set_page_config(page_title="SportsOracle Trending Topics Dashboard", layout="wide")
st.title("SportsOracle Trending Topics Dashboard")

trends = load_trends()

if not trends:
    st.warning("No trends data found. Please run the pipeline to generate outputs/trends_summary.json.")
    st.stop()

# --- Sidebar: Sport filter and refresh ---
st.sidebar.header("Filters")
category_options = ["All", "nba", "soccer"]
selected_sport = st.sidebar.selectbox("Select sport", category_options, index=0)
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

# --- Search box ---
search_query = st.text_input("Search summaries or titles", "")

# --- Filter trends by category ---
def filter_by_category(trends, selected_sport):
    # Drop cluster -1 (noise)
    trends = [t for t in trends if t["cluster_id"]] #!= -1]
    if selected_sport == "All":
        return [t for t in trends if t["category"] in ("nba", "soccer")]
    else:
        return [t for t in trends if t["category"] == selected_sport]

# TODO: Implement cluster relevancy/prioritization sorting (e.g., by recency, engagement, or a composite score)
filtered_trends = filter_by_category(trends, selected_sport)

# --- Search filter ---
def search_trends(trends, query):
    if not query.strip():
        return trends
    query = query.lower()
    results = []
    for t in trends:
        if query in t["summary"].lower() or any(query in title.lower() for title in t["top_titles"]):
            results.append(t)
    return results

filtered_trends = search_trends(filtered_trends, search_query)

# --- Main area: Show trend cards ---
if not filtered_trends:
    st.info("No trends match your filter/search.")
else:
    for trend in filtered_trends:
        display_title = pick_best_title(trend['summary'], trend['top_titles'])
        with st.container():
            st.subheader(f"{display_title}")  # Only show the title, no post count
            st.markdown(f"**Category:** {trend['category'].capitalize()}")
            st.markdown(f"**Keywords:** {' | '.join(trend['keywords'])}")
            st.markdown(f"**Summary:** {trend['summary']}")
            st.markdown("**Top Titles:**")
            # Exclude the display_title from the list of top titles
            for title in trend["top_titles"]:
                if title != display_title:
                    st.markdown(f"- {title}")
            st.markdown("---")
