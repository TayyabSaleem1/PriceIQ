import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_DIR
from models.elasticity import ElasticityModel
from models.demand_forecaster import DemandForecaster
from models.optimizer import PriceOptimizer

st.set_page_config(page_title="PriceIQ Dashboard", layout="wide")

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
    else:
        # We need a trained elasticity model for the optimizer to work well.
        # But if it's not trained, we can just use an empty one.
        pass
    # Demand forecaster is instantiated per product, so we just pass None for now
    # or the optimizer handles base demand directly.
    return PriceOptimizer(elasticity_model=elast_model, demand_forecaster=None)

@st.cache_data(ttl=3600)
def get_recommendations(_optimizer, fs_df):
    # Get latest data per product
    latest_df = fs_df.sort_values('date').groupby('product_id').last().reset_index()
    latest_df = latest_df.drop(columns=['competitor_price']).rename(columns={'units_sold': 'base_demand', 'price_charged': 'current_price', 'avg_competitor_price': 'competitor_price', 'current_stock': 'stock_level'}).reset_index(drop=True)
    
    # Fill any missing
    if 'competitor_price' not in latest_df.columns:
        latest_df['competitor_price'] = latest_df['current_price'] * 1.05
    
    rec_df = _optimizer.optimize_portfolio(latest_df)
    return rec_df, latest_df

def main():
    st.title("📊 Live Price Intelligence Dashboard")
    
    fs_df = load_data()
    if fs_df is None:
        st.warning("Data not found. Run ETL pipeline first.")
        return
        
    optimizer = get_optimizer()
    rec_df, latest_df = get_recommendations(optimizer, fs_df)
    
    # Metrics
    avg_price_gap = ((latest_df['current_price'] - latest_df['competitor_price']) / latest_df['competitor_price'] * 100).mean()
    overpriced = len(latest_df[latest_df['current_price'] > latest_df['competitor_price']])
    underpriced = len(latest_df[latest_df['current_price'] < latest_df['competitor_price']])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Price Gap vs Competitors (%)", f"{avg_price_gap:.1f}%")
    c2.metric("Products Overpriced vs Competitors", overpriced)
    c3.metric("Products Underpriced vs Competitors", underpriced)
    
    st.markdown("---")
    
    # Section 1 - Price Tracker Table
    st.subheader("Price Tracker")
    tracker_df = rec_df[['product_id', 'category', 'current_price', 'optimal_price', 'price_change_pct']].copy()
    
    # Merge avg competitor price
    tracker_df = pd.merge(tracker_df, latest_df[['product_id', 'competitor_price']], on='product_id', how='left')
    tracker_df = tracker_df.rename(columns={'competitor_price': 'Avg Competitor Price'})
    
    tracker_df = tracker_df[['product_id', 'category', 'current_price', 'optimal_price', 'Avg Competitor Price', 'price_change_pct']]
    tracker_df.columns = ['Product', 'Category', 'Current Price', 'Optimal Price', 'Avg Competitor Price', 'Recommended Change %']
    
    def color_change(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'background-color: {color}'
        
    styled_df = tracker_df.style.map(color_change, subset=['Recommended Change %']).format(
        {'Current Price': '${:.2f}', 'Optimal Price': '${:.2f}', 'Avg Competitor Price': '${:.2f}', 'Recommended Change %': '{:.1f}%'}
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # Section 2 - Margin Heatmap
    st.subheader("Margin Heatmap by Product and Category")
    top_categories = latest_df['category'].value_counts().nlargest(5).index
    heatmap_data = latest_df[latest_df['category'].isin(top_categories)]
    
    # Create pivot table
    pivot_df = heatmap_data.pivot_table(index='category', columns='product_id', values='margin', aggfunc='mean')
    # Limit to top 10 products for clarity
    top_10_products = heatmap_data.sort_values('margin', ascending=False)['product_id'].head(10).values
    pivot_df = pivot_df[[col for col in pivot_df.columns if col in top_10_products]]
    
    fig_heatmap = px.imshow(pivot_df, color_continuous_scale="RdYlGn", title="Margin Heatmap by Product and Category",
                            labels=dict(x="Product", y="Category", color="Margin %"))
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Section 3 - Demand Forecast Chart
    st.subheader("7-Day Demand Forecast with Confidence Interval")
    
    selected_prod = st.selectbox("Select Product", fs_df['product_id'].unique())
    prod_sales = fs_df[fs_df['product_id'] == selected_prod].copy()
    
    if len(prod_sales) > 30: # Need enough data
        import threading
        
        forecast_result = {}
        
        def run_forecast():
            try:
                f = DemandForecaster(product_id=selected_prod, use_xgboost_residuals=False)
                f.fit(prod_sales)
                forecast_result['df'] = f.predict(horizon_days=7)
            except Exception as e:
                forecast_result['error'] = str(e)
        
        with st.spinner("Generating forecast..."):
            t = threading.Thread(target=run_forecast)
            t.start()
            t.join(timeout=60)
        
        if 'error' in forecast_result:
            st.error(f"Forecast failed: {forecast_result['error']}")
        elif 'df' not in forecast_result:
            st.warning("Forecast timed out. Try a different product.")
        else:
            forecast_df = forecast_result['df']
            
            # Historical plot
            hist_plot = prod_sales.tail(30)
            hist_plot['ds'] = pd.to_datetime(hist_plot['date'])
            
            fig_forecast = go.Figure()
            
            # Historical
            fig_forecast.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['units_sold'], mode='lines', name='Historical Sales', line=dict(color='blue', width=2)))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['final_forecast'], mode='lines', name='Forecast', line=dict(color='orange', width=2, dash='dash')))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Band'
            ))
            
            today = pd.to_datetime(hist_plot['ds'].max())
            fig_forecast.add_vline(x=today, line_width=2, line_dash="dash", line_color="gray", annotation_text="Today")
            
            fig_forecast.update_layout(title="7-Day Demand Forecast with Confidence Interval", xaxis_title="Date", yaxis_title="Units Sold")
            st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Not enough data to generate forecast.")
        
    st.markdown("---")
    
    # Section 4 - Price Distribution
    st.subheader("Our Price vs. Competitor Average by Category")
    cat_price_df = latest_df.groupby('category')[['current_price', 'competitor_price']].mean().reset_index(drop=True)
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=cat_price_df['category'], y=cat_price_df['current_price'], name='Our Avg Price', marker_color='blue'))
    fig_bar.add_trace(go.Bar(x=cat_price_df['category'], y=cat_price_df['competitor_price'], name='Competitor Avg Price', marker_color='gray'))
    
    fig_bar.update_layout(title="Our Price vs. Competitor Average by Category", barmode='group', xaxis_title="Category", yaxis_title="Price ($)")
    st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()


