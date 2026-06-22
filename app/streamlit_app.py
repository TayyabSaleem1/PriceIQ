import sys
import os
import streamlit as st
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

st.set_page_config(page_title="PriceIQ", page_icon="💹", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(ttl=3600)
def check_data_ready():
    fs_path = PROCESSED_DIR / "feature_store.csv"
    return fs_path.exists()

def main():
    st.sidebar.markdown("## 💹 PriceIQ")
    st.sidebar.caption("Dynamic Pricing Intelligence")
    
    data_ready = check_data_ready()
    if data_ready:
        st.sidebar.success("🟢 Data Ready")
    else:
        st.sidebar.error("🔴 Run ETL Pipeline First")
        
    st.sidebar.info("Use the sidebar pages to navigate")
    
    # Initialize session state
    st.session_state.setdefault("selected_product_id", None)
    st.session_state.setdefault("rationale_cache", {})
    
    st.title("PriceIQ — Pricing Intelligence Dashboard")
    
    if not data_ready:
        st.warning("Data not found. Please run the ETL pipeline first:\n`python pipeline/etl.py`")
        return
        
    # Example KPI Cards (in a real app, these would be aggregated from the feature store and optimizer results)
    # We will load actual data in the pages, but for the main page we can load just enough to show KPIs
    try:
        fs_df = pd.read_csv(PROCESSED_DIR / "feature_store.csv")
        num_products = fs_df['product_id'].nunique()
        # Mocking some aggregated stats for the home page since optimization happens per page or pre-computed
        avg_change = 4.2 # Mock, usually would come from optimizer_portfolio output
        uplift = 12500 # Mock
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products Tracked", f"{num_products}")
        col2.metric("Avg Recommended Price Change %", f"+{avg_change}%")
        col3.metric("Estimated Revenue Uplift $", f"${uplift:,}")
        col4.metric("Last Updated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        
        st.markdown("### Welcome to PriceIQ")
        st.markdown("""
        Select a page from the sidebar to begin:
        - **Dashboard**: Live price tracking, margins, and demand forecasting.
        - **Recommendations**: AI-driven price recommendations and Claude-generated rationale.
        - **What-If Simulator**: Interactive price elasticity modeling.
        - **Competitor Intel**: Sentiment analysis and competitor weakness scoring.
        """)
    except Exception as e:
        st.error(f"Error loading dashboard metrics: {e}")

if __name__ == "__main__":
    main()
