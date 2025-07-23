import streamlit as st
import json

# Load trends data
@st.cache_data
def load_trends():
    with open("outputs/trends_summary.json") as f:
        return json.load(f)

trends = load_trends()

# Sidebar: Sport filter (example using keywords)
all_keywords = set(kw for trend in trends for kw in trend["keywords"])
sport = st.sidebar.selectbox("Filter by sport keyword", sorted(all_keywords))

# Main: Trend cards
for trend in trends:
    if sport and sport not in trend["keywords"]:
        continue
    st.subheader(f"Trend {trend['cluster_id']}")
    st.write("**Summary:**", trend["summary"])
    st.write("**Keywords:**", ", ".join(trend["keywords"]))
    st.write("**Top Titles:**")
    for title in trend["top_titles"]:
        st.write("-", title)
    st.markdown("---")

# Search box
query = st.text_input("Search posts")
if query:
    # Call your FAISS search function here and display results
    pass

# Refresh button
if st.button("Refresh Data"):
    st.experimental_rerun()