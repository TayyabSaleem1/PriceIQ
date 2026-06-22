import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COST_MARGIN_FLOOR

class PriceOptimizer:
    def __init__(self, elasticity_model, demand_forecaster):
        self.elasticity_model = elasticity_model
        self.demand_forecaster = demand_forecaster

    def _revenue_objective(self, price_array, base_demand, elasticity, cost_price, base_price):
        price = price_array[0]
        
        # Demand calculation based on elasticity
        # log(D) - log(D0) = E * (log(P) - log(P0)) is the strictly correct way
        # But instructions ask for linear approximation: demand = base_demand * (1 + elasticity * ((price - base_price) / base_price))
        price_change_pct = (price - base_price) / base_price
        demand = base_demand * (1 + elasticity * price_change_pct)
        demand = max(demand, 0)
        
        margin = (price - cost_price) / price
        
        # We want to minimize negative revenue (i.e. maximize revenue)
        revenue = price * demand
        return -revenue

    def optimize(self, product_id, cost_price, current_price, category, base_demand, competitor_price, stock_level):
        price_floor = cost_price * (1 + COST_MARGIN_FLOOR)
        price_ceiling = competitor_price * 1.2
        
        # Edge cases
        if stock_level == 0:
            return {
                "optimal_price": current_price,
                "expected_revenue": current_price * base_demand,
                "expected_demand": base_demand,
                "expected_margin": (current_price - cost_price) / current_price,
                "price_change_pct": 0.0,
                "confidence": "low",
                "reason": "out of stock",
                "price_floor": price_floor,
                "price_ceiling": price_ceiling
            }
            
        elasticity_metrics = self.elasticity_model.elasticities.get(category, {})
        elasticity = elasticity_metrics.get('elasticity', -1.0)
        r2 = elasticity_metrics.get('r2', 0.0)
        
        # Setup scipy optimization
        initial_guess = [float(current_price)]
        bounds = [(price_floor, price_ceiling)]
        
        # Margin constraint: (price - cost_price)/price >= COST_MARGIN_FLOOR => price*(1-COST_MARGIN_FLOOR) >= cost_price
        # Actually handled by price_floor since price >= cost_price*(1+COST_MARGIN_FLOOR) > cost_price / (1-COST_MARGIN_FLOOR) roughly.
        # Strict constraint function for SLSQP:
        def margin_constraint(x):
            return (x[0] - cost_price) / x[0] - COST_MARGIN_FLOOR
            
        constraints = [{'type': 'ineq', 'fun': margin_constraint}]
        
        result = minimize(
            self._revenue_objective,
            initial_guess,
            args=(base_demand, elasticity, cost_price, current_price),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_price = result.x[0]
        price_change_pct = (optimal_price - current_price) / current_price
        expected_demand = max(0, base_demand * (1 + elasticity * price_change_pct))
        expected_revenue = optimal_price * expected_demand
        expected_margin = (optimal_price - cost_price) / optimal_price
        
        # Confidence logic
        if result.success and r2 > 0.5:
            confidence = "high"
        elif result.success and r2 <= 0.5:
            confidence = "medium"
        else:
            confidence = "low"
            
        return {
            "optimal_price": optimal_price,
            "expected_revenue": expected_revenue,
            "expected_demand": expected_demand,
            "expected_margin": expected_margin,
            "price_change_pct": price_change_pct * 100, # returned as percentage value
            "confidence": confidence,
            "price_floor": price_floor,
            "price_ceiling": price_ceiling
        }

    def optimize_portfolio(self, products_df):
        results = []
        for row in products_df.itertuples(index=False):
            try:
                result = self.optimize(
                    product_id=str(row.product_id),
                    cost_price=float(row.cost_price),
                    current_price=float(row.current_price),
                    category=str(row.category),
                    base_demand=float(row.base_demand),
                    competitor_price=float(row.competitor_price),
                    stock_level=int(row.stock_level)
                )
                result['product_id'] = str(row.product_id)
                result['product_name'] = str(row.product_name) if hasattr(row, 'product_name') else str(row.product_id)
                result['category'] = str(row.category)
                result['current_price'] = float(row.current_price)
                result['current_revenue'] = float(row.current_price) * float(row.base_demand)
                results.append(result)
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        rec_df = pd.DataFrame(results)
        rec_df['revenue_uplift_pct'] = (
            (rec_df['expected_revenue'] - rec_df['current_revenue']) / rec_df['current_revenue'] * 100
        )
        return rec_df.sort_values('revenue_uplift_pct', ascending=False).reset_index(drop=True)
