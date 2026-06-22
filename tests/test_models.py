import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.elasticity import ElasticityModel
from models.optimizer import PriceOptimizer
from models.sentiment import SentimentAnalyzer
from models.demand_forecaster import DemandForecaster
from llm.rationale_generator import RationaleGenerator
from data.synthetic.generate_synthetic import generate_inventory, generate_sales_history
from config import COST_MARGIN_FLOOR

def test_elasticity_fit():
    np.random.seed(42)
    # Higher price = lower qty
    prices = np.random.uniform(10, 100, 100)
    qty = 1000 / prices + np.random.normal(0, 5, 100)
    qty = np.maximum(qty, 1) # No negative or zero qty
    
    df = pd.DataFrame({
        'category': ['test_cat'] * 100,
        'price': prices,
        'qty': qty
    })
    
    model = ElasticityModel()
    res = model.fit(df, 'category', 'price', 'qty')
    
    assert 'test_cat' in res
    elasticity = res['test_cat']['elasticity']
    r2 = res['test_cat']['r2']
    
    assert elasticity < 0  # Normal good
    assert 0 <= r2 <= 1.0

def test_optimizer_respects_margin_floor():
    class MockElasticity:
        def __init__(self):
            self.elasticities = {'test_cat': {'elasticity': -1.5, 'r2': 0.8}}
        def get_elasticity(self, cat):
            return -1.5
            
    mock_elasticity = MockElasticity()
    optimizer = PriceOptimizer(elasticity_model=mock_elasticity, demand_forecaster=None)
    
    cost_price = 100.0
    current_price = 150.0
    
    res = optimizer.optimize("P001", cost_price, current_price, "test_cat", 100, 200.0, 50)
    
    assert res["optimal_price"] >= cost_price * (1 + COST_MARGIN_FLOOR)

def test_optimizer_bounds():
    class MockElasticity:
        def __init__(self):
            self.elasticities = {'test_cat': {'elasticity': -1.5, 'r2': 0.8}}
        def get_elasticity(self, cat):
            return -1.5
            
    optimizer = PriceOptimizer(elasticity_model=MockElasticity(), demand_forecaster=None)
    
    cost_price = 10.0
    competitor_price = 20.0
    current_price = 15.0
    
    res = optimizer.optimize("P002", cost_price, current_price, "test_cat", 50, competitor_price, 100)
    
    optimal_price = res["optimal_price"]
    assert optimal_price <= competitor_price * 1.2
    assert optimal_price >= cost_price * (1 + COST_MARGIN_FLOOR)

def test_sentiment_analyzer_basic():
    analyzer = SentimentAnalyzer(model_name="distilbert-base-uncased-finetuned-sst-2-english")
    texts = ["This product is amazing and high quality", "Terrible, broke after one day", "It is okay"]
    results = analyzer.analyze_batch(texts)
    
    assert results[0]['label'] == "POSITIVE"
    assert results[1]['label'] == "NEGATIVE"
    
def test_etl_synthetic_data_shape():
    inventory_df = generate_inventory(num_products=50)
    assert len(inventory_df) == 50
    assert 'product_id' in inventory_df.columns
    assert 'cost_price' in inventory_df.columns
    
    sales_df = generate_sales_history(inventory_df, start_date="2023-01-01", end_date="2024-12-31")
    # 2 years = 730 days
    assert len(sales_df) >= 50 * 730

def test_rationale_prompt_structure():
    rg = RationaleGenerator(api_key="dummy")
    
    prompt = rg.build_prompt(
        product_name="Test Widget",
        category="electronics",
        current_price=10.0,
        optimal_price=12.0,
        price_change_pct=20.0,
        expected_demand_change=-5.0,
        elasticity=-1.5,
        forecast_7d=100.0,
        competitor_weakness_score=0.8,
        sentiment_summary="Mixed",
        margin=0.2,
        confidence="high"
    )
    
    assert "system" in prompt
    assert "user" in prompt
    assert "Test Widget" in prompt["user"]

def test_demand_forecaster_output_shape():
    from unittest.mock import patch
    # Create 90 days of synthetic sales for one product
    dates = pd.date_range(start="2024-01-01", periods=90)
    df = pd.DataFrame({
        'date': dates,
        'product_id': ['P001'] * 90,
        'units_sold': np.random.poisson(20, 90),
        'promo_flag': np.random.choice([0, 1], 90, p=[0.9, 0.1]),
        'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates]
    })
    
    with patch('models.demand_forecaster.Prophet') as MockProphet:
        mock_instance = MockProphet.return_value
        
        def mock_make_future(periods):
            future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=periods)
            return pd.DataFrame({'ds': future_dates})
        mock_instance.make_future_dataframe.side_effect = mock_make_future
        
        def mock_predict(future_df):
            res = future_df.copy()
            res['yhat'] = 25.0
            res['yhat_lower'] = 20.0
            res['yhat_upper'] = 30.0
            return res
        mock_instance.predict.side_effect = mock_predict
        
        forecaster = DemandForecaster(product_id="P001", use_xgboost_residuals=False)
        forecaster.fit(df)
        preds = forecaster.predict(horizon_days=7)
    
    assert len(preds) == 7
    assert 'ds' in preds.columns
    assert 'final_forecast' in preds.columns
    assert 'lower_bound' in preds.columns
    assert 'upper_bound' in preds.columns
