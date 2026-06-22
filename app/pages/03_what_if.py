import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_DIR, COST_MARGIN_FLOOR
from models.elasticity import ElasticityModel
from models.optimizer import PriceOptimizer

st.set_page_config(page_title="PriceIQ What-If Simulator", layout="wide")

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

def main():
    st.title("🔬 What-If Price Simulator")
    
    fs_df = load_data()
    if fs_df is None:
        st.warning("Data not found. Run ETL pipeline first.")
        return
        
    latest_df = fs_df.sort_values('date').groupby('product_id').last().reset_index()
    latest_df = latest_df.drop(columns=['competitor_price']).rename(columns={'units_sold': 'base_demand', 'price_charged': 'current_price', 'avg_competitor_price': 'competitor_price', 'current_stock': 'stock_level'})
    if 'competitor_price' not in latest_df.columns:
        latest_df['competitor_price'] = latest_df['current_price'] * 1.05
        
    optimizer = get_optimizer()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        product_names = latest_df['product_id'] + " - " + latest_df['product_name']
        selected_prod_name = st.selectbox("Select Product", product_names)
        
        prod_id = selected_prod_name.split(" - ")[0]
        prod_data = latest_df[latest_df['product_id'] == prod_id].iloc[0]
        
        cost_price = float(prod_data['cost_price'])
        current_price = float(prod_data['current_price'])
        competitor_price = float(prod_data['competitor_price'])
        base_demand = float(prod_data['base_demand'])
        category = str(prod_data['category'])
        
        # Get optimizer result for reference
        opt_res = optimizer.optimize(prod_id, cost_price, current_price, category, base_demand, competitor_price, prod_data['stock_level'])
        
        price_floor = cost_price * (1 + COST_MARGIN_FLOOR)
        price_ceiling = competitor_price * 1.5
        
        # Reactive slider
        test_price = st.slider(
            "Simulated Price ($)", 
            min_value=float(cost_price), 
            max_value=float(price_ceiling), 
            step=0.50, 
            value=float(current_price)
        )
        
        st.caption(f"Cost: ${cost_price:.2f} | Floor: ${price_floor:.2f} | Ceiling: ${price_ceiling:.2f}")
        
    with col2:
        # Calculate simulated metrics
        elasticity = optimizer.elasticity_model.get_elasticity(category)
        price_change_pct = (test_price - current_price) / current_price
        
        sim_demand = max(0, base_demand * (1 + elasticity * price_change_pct))
        sim_revenue = test_price * sim_demand
        sim_margin = (test_price - cost_price) / test_price
        
        current_revenue = current_price * base_demand
        current_margin = (current_price - cost_price) / current_price
        
        st.subheader("Section 1 — Live Output")
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Expected Units Sold", f"{sim_demand:.1f}", f"{sim_demand - base_demand:.1f} vs current")
        c2.metric("Expected Revenue $", f"${sim_revenue:.2f}", f"${sim_revenue - current_revenue:.2f} vs current")
        c3.metric("Expected Margin %", f"{sim_margin*100:.1f}%", f"{(sim_margin - current_margin)*100:.1f}% vs current")
        
        gap_pct = (test_price - competitor_price) / competitor_price * 100
        gap_text = "above" if gap_pct > 0 else "below"
        st.info(f"You are {abs(gap_pct):.1f}% {gap_text} competitor avg (${competitor_price:.2f})")
        
        st.markdown("---")
        
        st.subheader("Section 2 — Scenario Table")
        scenarios = []
        for pct in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            p = current_price * (1 + pct)
            d = max(0, base_demand * (1 + elasticity * pct))
            r = p * d
            m = (p - cost_price) / p
            scenarios.append({
                "Scenario": f"{pct*100:+.0f}%" if pct != 0 else "Current",
                "Price": p,
                "Units Sold": d,
                "Revenue": r,
                "Margin %": m * 100
            })
            
        scenarios_df = pd.DataFrame(scenarios)
        
        # Highlight optimizer recommendation row
        def highlight_optimal(row):
            if abs(row['Price'] - opt_res['optimal_price']) < 0.1:
                return ['border: 2px solid gold'] * len(row)
            return [''] * len(row)
            
        styled_scenarios = scenarios_df.style.apply(highlight_optimal, axis=1).format({
            "Price": "${:.2f}",
            "Units Sold": "{:.1f}",
            "Revenue": "${:.2f}",
            "Margin %": "{:.1f}%"
        })
        st.dataframe(styled_scenarios, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Section 3 — Price-Demand Curve")
        prices = np.linspace(cost_price, price_ceiling, 50)
        demands = [max(0, base_demand * (1 + elasticity * ((p - current_price)/current_price))) for p in prices]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=demands, mode='lines', name='Demand Curve'))
        
        fig.add_vline(x=current_price, line_dash="dash", line_color="blue", annotation_text="Current")
        fig.add_vline(x=opt_res['optimal_price'], line_dash="solid", line_color="green", annotation_text="Optimal")
        fig.add_vline(x=competitor_price, line_dash="dash", line_color="orange", annotation_text="Competitor")
        
        # Shaded region below cost floor
        fig.add_vrect(x0=cost_price, x1=price_floor, fillcolor="red", opacity=0.2, layer="below", annotation_text="Below Margin Floor")
        
        fig.update_layout(title="Price vs. Expected Demand Curve", xaxis_title="Price ($)", yaxis_title="Expected Demand")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Section 4 — A/B Test Simulator")
        col_dur, col_split = st.columns(2)
        test_duration = col_dur.number_input("Test Duration (days)", min_value=1, max_value=90, value=14)
        traffic_split = col_split.slider("Traffic Split % (Treatment)", 10, 90, 50)
        
        treat_traffic = traffic_split / 100
        ctrl_traffic = 1.0 - treat_traffic
        
        daily_control_rev = current_price * (base_demand * ctrl_traffic)
        daily_treat_rev = opt_res['optimal_price'] * (opt_res['expected_demand'] * treat_traffic)
        
        days = np.arange(1, test_duration + 1)
        ctrl_cumulative = daily_control_rev * days
        treat_cumulative = daily_treat_rev * days
        
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Scatter(x=days, y=ctrl_cumulative, mode='lines', name='Control Group (Current)'))
        fig_ab.add_trace(go.Scatter(x=days, y=treat_cumulative, mode='lines', name='Treatment Group (Recommended)'))
        fig_ab.update_layout(title="Projected Cumulative Revenue", xaxis_title="Day", yaxis_title="Revenue ($)")
        
        st.plotly_chart(fig_ab, use_container_width=True)
        st.caption("Note: Projection based on elasticity model. Actual results depend on traffic and market conditions.")

if __name__ == "__main__":
    main()

