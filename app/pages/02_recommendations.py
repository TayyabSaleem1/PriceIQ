import sys
import os
import streamlit as st
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_DIR, PRODUCT_CATEGORIES
from models.elasticity import ElasticityModel
from models.optimizer import PriceOptimizer
from llm.rationale_generator import RationaleGenerator

st.set_page_config(page_title="PriceIQ Recommendations", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    fs_path = PROCESSED_DIR / "feature_store.csv"
    if not fs_path.exists():
        return None
    return pd.read_csv(fs_path)

@st.cache_resource
def get_optimizer():
    elast_model = ElasticityModel()
    elast_path = PROCESSED_DIR / "elasticity_model.pkl"
    if elast_path.exists():
        elast_model.load(elast_path)
    return PriceOptimizer(elasticity_model=elast_model, demand_forecaster=None)

@st.cache_data(ttl=3600)
def get_recommendations(_optimizer, fs_df):
    latest_df = fs_df.sort_values('date').groupby('product_id').last().reset_index()
    latest_df = latest_df.rename(columns={'units_sold': 'base_demand', 'price_charged': 'current_price', 'avg_competitor_price': 'competitor_price', 'current_stock': 'stock_level'})
    
    if 'competitor_price' not in latest_df.columns:
        latest_df['competitor_price'] = latest_df['current_price'] * 1.05
    
    rec_df = _optimizer.optimize_portfolio(latest_df)
    
    return rec_df

def generate_fallback_rationale(product_name, current_price, optimal_price, 
                                 price_change_pct, confidence, category):
    direction = "increase" if price_change_pct > 0 else "decrease"
    magnitude = abs(price_change_pct)
    return (
        f"Analysis recommends a {magnitude:.1f}% {direction} in price for "
        f"{product_name} from ${current_price:.2f} to ${optimal_price:.2f}. "
        f"This {category} product is priced relative to competitor benchmarks "
        f"with {confidence} confidence based on elasticity modeling. "
        f"Monitor units sold over the next 7 days to validate demand response."
    )

def get_prompt_inputs(row):
    return {
        "product_name": row['product_name'],
        "category": row['category'],
        "current_price": row['current_price'],
        "optimal_price": row['optimal_price'],
        "price_change_pct": row['price_change_pct'],
        "expected_demand_change": row['expected_demand'] - (row['expected_demand'] / (1 + (row['price_change_pct']/100) * -1.0) if row['price_change_pct'] != 0 else row['expected_demand']),
        "elasticity": -1.0,
        "forecast_7d": row['expected_demand'] * 7,
        "competitor_weakness_score": 0.5,
        "sentiment_summary": "Mixed reviews, pricing cited as a concern.",
        "margin": row['expected_margin'],
        "confidence": row['confidence']
    }

def main():
    st.title("🤖 AI Pricing Recommendations")
    
    fs_df = load_data()
    if fs_df is None:
        st.warning("Data not found. Run ETL pipeline first.")
        return
        
    optimizer = get_optimizer()
    rec_df = get_recommendations(optimizer, fs_df)
    
    if 'rationale_cache' not in st.session_state:
        st.session_state.rationale_cache = {}
        
    # Sidebar filters
    st.sidebar.subheader("Filters")
    selected_categories = st.sidebar.multiselect("Category", options=PRODUCT_CATEGORIES, default=PRODUCT_CATEGORIES)
    min_confidence = st.sidebar.selectbox("Min Confidence", ["High", "Medium", "Low"])
    sort_by = st.sidebar.selectbox("Sort By", ["Revenue Uplift %", "Price Change %", "Margin"])
    
    # Apply filters
    filtered_df = rec_df[rec_df['category'].isin(selected_categories)]
    
    conf_map = {"Low": ["high", "medium", "low"], "Medium": ["high", "medium"], "High": ["high"]}
    filtered_df = filtered_df[filtered_df['confidence'].isin(conf_map[min_confidence])]
    
    sort_map = {"Revenue Uplift %": "revenue_uplift_pct", "Price Change %": "price_change_pct", "Margin": "expected_margin"}
    filtered_df = filtered_df.sort_values(sort_map[sort_by], ascending=False)
    
    # Batch Generate
    if st.button("Generate AI Rationale for Top 5"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        top_5 = filtered_df.head(5)
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            if row['confidence'] in ['high', 'medium'] and row['product_id'] not in st.session_state.rationale_cache:
                if not api_key:
                    rationale = generate_fallback_rationale(
                        row['product_name'], row['current_price'], row['optimal_price'], 
                        row['price_change_pct'], row['confidence'], row['category']
                    )
                    st.session_state.rationale_cache[row['product_id']] = {
                        'rationale': rationale,
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model': 'Rule-based Fallback'
                    }
                else:
                    generator = RationaleGenerator(api_key=api_key)
                    prompt_inputs = get_prompt_inputs(row)
                    result = generator.generate(prompt_inputs)
                    result['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.rationale_cache[row['product_id']] = result
            progress_bar.progress((i + 1) / 5)
        st.success("Batch generation complete!")
        
    st.markdown("---")
    
    for _, row in filtered_df.iterrows():
        with st.container():
            st.markdown(f"### {row['product_name']} <span style='font-size: 0.6em; background-color: #333; color: white; padding: 2px 6px; border-radius: 4px;'>{row['category']}</span>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
            
            # Price arrow logic
            price_color = "green" if row['optimal_price'] > row['current_price'] else "red" if row['optimal_price'] < row['current_price'] else "gray"
            arrow = "↑" if row['optimal_price'] > row['current_price'] else "↓" if row['optimal_price'] < row['current_price'] else "→"
            
            c1.markdown(f"**Price:** ${row['current_price']:.2f} ➔ <span style='color: {price_color}'>**${row['optimal_price']:.2f}** ({arrow}{abs(row['price_change_pct']):.1f}%)</span>", unsafe_allow_html=True)
            c2.markdown(f"**Uplift:** {row['revenue_uplift_pct']:.1f}%")
            
            conf_color = "🟢" if row['confidence'] == 'high' else "🟡" if row['confidence'] == 'medium' else "🔴"
            c3.markdown(f"{conf_color} {row['confidence'].capitalize()} Conf")
            
            with c4:
                btn_key = f"btn_{row['product_id']}"
                if st.button("Generate AI Rationale", key=btn_key):
                    if row['confidence'] == 'low':
                        st.warning("Confidence too low for AI rationale. Review manually.")
                    else:
                        api_key = os.getenv("ANTHROPIC_API_KEY")
                        if not api_key:
                            rationale = generate_fallback_rationale(
                                row['product_name'], row['current_price'], row['optimal_price'], 
                                row['price_change_pct'], row['confidence'], row['category']
                            )
                            result = {
                                'rationale': rationale,
                                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'model': 'Rule-based Fallback'
                            }
                            st.session_state.rationale_cache[row['product_id']] = result
                            st.info(result['rationale'])
                        else:
                            generator = RationaleGenerator(api_key=api_key)
                            with st.spinner("Analyzing pricing signals..."):
                                prompt_inputs = get_prompt_inputs(row)
                                result = generator.generate(prompt_inputs)
                                result['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.rationale_cache[row['product_id']] = result
                                st.info(result['rationale'])
            
            if row['product_id'] in st.session_state.rationale_cache:
                cache_data = st.session_state.rationale_cache[row['product_id']]
                st.info(f"*{cache_data['rationale']}*")
                st.caption(f"Model: {cache_data.get('model', 'Unknown')} | Tokens: {cache_data.get('tokens_used', 0)} | Generated: {cache_data.get('timestamp', '')}")
            
            st.markdown("---")

if __name__ == "__main__":
    main()
