import os
import sys
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ELASTICITY_MIN_SAMPLES, PROCESSED_DIR

class ElasticityModel:
    def __init__(self):
        self.elasticities = {}

    def fit(self, df, category_col, price_col, qty_col):
        """
        Fits OLS regression: log(qty) ~ log(price) + log(competitor_price) + promo_flag
        using statsmodels.
        """
        results = {}
        
        # Ensure we don't have zeros or negative values for log
        df_clean = df[
            (df[price_col] > 0) & 
            (df[qty_col] > 0)
        ].copy()
        
        df_clean['log_qty'] = np.log(df_clean[qty_col])
        df_clean['log_price'] = np.log(df_clean[price_col])
        
        if 'competitor_price' in df_clean.columns:
            df_clean = df_clean[df_clean['competitor_price'] > 0]
            df_clean['log_comp_price'] = np.log(df_clean['competitor_price'])
        
        if 'promo_flag' not in df_clean.columns:
            df_clean['promo_flag'] = 0
            
        for category, group in df_clean.groupby(category_col):
            if len(group) >= ELASTICITY_MIN_SAMPLES:
                y = group['log_qty']
                
                features = ['log_price', 'promo_flag']
                if 'log_comp_price' in group.columns:
                    features.append('log_comp_price')
                    
                X = group[features]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                
                elasticity = model.params.get('log_price', 0)
                r2 = model.rsquared
                p_value = model.pvalues.get('log_price', 1.0)
                
                results[category] = {
                    'elasticity': elasticity,
                    'r2': r2,
                    'p_value': p_value,
                    'n': len(group)
                }
                
        self.elasticities = results
        return results

    def get_elasticity(self, category):
        if category in self.elasticities:
            return self.elasticities[category]['elasticity']
        
        # Fallback to mean elasticity if category not found
        if self.elasticities:
            return np.mean([v['elasticity'] for v in self.elasticities.values()])
        return -1.0  # Default assumed elasticity

    def predict_demand_change(self, category, price_change_pct):
        elasticity = self.get_elasticity(category)
        return elasticity * price_change_pct

    def summary_table(self):
        rows = []
        for cat, metrics in self.elasticities.items():
            interpretation = "Elastic" if abs(metrics['elasticity']) > 1 else "Inelastic"
            rows.append({
                'category': cat,
                'elasticity': metrics['elasticity'],
                'r2': metrics['r2'],
                'n_samples': metrics['n'],
                'interpretation': interpretation
            })
            
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by='elasticity', key=abs, ascending=False).reset_index(drop=True)
        return df

    def save(self, filepath=None):
        if filepath is None:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            filepath = PROCESSED_DIR / "elasticity_model.pkl"
        joblib.dump(self.elasticities, filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = PROCESSED_DIR / "elasticity_model.pkl"
        self.elasticities = joblib.load(filepath)
