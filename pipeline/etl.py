import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceIQETL:
    def load_retail_price_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            
            # Clean: drop nulls in essential columns
            df = df.dropna(subset=['unit_price', 'qty'])
            
            # Clip price outliers per category
            if 'product_category_name' in df.columns and 'unit_price' in df.columns:
                def clip_outliers(group):
                    low = group['unit_price'].quantile(0.01)
                    high = group['unit_price'].quantile(0.99)
                    group['unit_price'] = group['unit_price'].clip(lower=low, upper=high)
                    return group
                
                df = df.groupby('product_category_name', group_keys=False).apply(clip_outliers)
            
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}. Please download retail_price.csv and place it in {RAW_DIR}")
            raise

    def load_amazon_reviews(self, filepath, sample_n=50000):
        try:
            df = pd.read_csv(filepath)
            
            if len(df) > sample_n:
                df = df.sample(n=sample_n, random_state=42)
                
            # Create sentiment_label
            df = df[df['Score'] != 3].copy()
            df['sentiment_label'] = np.where(df['Score'] >= 4, 'positive', 'negative')
            
            return df[['ProductId', 'Text', 'sentiment_label']]
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}. Please download Reviews.csv and place it in {RAW_DIR}")
            raise

    def load_synthetic_data(self):
        inventory_path = SYNTHETIC_DIR / "inventory.csv"
        sales_path = SYNTHETIC_DIR / "sales_history.csv"
        
        try:
            inventory_df = pd.read_csv(inventory_path)
            sales_df = pd.read_csv(sales_path)
            
            merged_df = pd.merge(sales_df, inventory_df, on="product_id", how="inner")
            merged_df['margin'] = (merged_df['price_charged'] - merged_df['cost_price']) / merged_df['price_charged']
            
            return merged_df
        except FileNotFoundError as e:
            logging.error(f"Synthetic data missing. Please run data/synthetic/generate_synthetic.py first. Details: {e}")
            raise

    def build_feature_store(self, retail_df, sales_df):
        # In a real app, we might merge these, but here they might not share common keys.
        # We will focus on building features on the synthetic sales_df as our primary feature store.
        df = sales_df.copy()
        
        if 'competitor_1_price' in df.columns:
            df['price_ratio'] = df['price_charged'] / df['competitor_1_price']
            
        if all(col in df.columns for col in ['competitor_1_price', 'competitor_2_price', 'competitor_3_price']):
            df['avg_competitor_price'] = df[['competitor_1_price', 'competitor_2_price', 'competitor_3_price']].mean(axis=1)
        elif 'competitor_price' in df.columns:
            df['avg_competitor_price'] = df['competitor_price']
            df['price_ratio'] = df['price_charged'] / df['competitor_price']
            
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = PROCESSED_DIR / "feature_store.csv"
        df.to_csv(out_path, index=False)
        return df

    def validate_data(self, df, required_cols, name):
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset {name} missing required columns: {missing_cols}")
            
        null_pcts = df[required_cols].isnull().mean()
        for col, pct in null_pcts.items():
            logging.info(f"{name} - {col} null %: {pct:.2%}")
            if pct > 0.5:
                raise ValueError(f"Column {col} in {name} has >50% nulls ({pct:.2%})")

    def run_full_pipeline(self):
        try:
            logging.info("Starting ETL pipeline...")
            
            logging.info("Loading synthetic data...")
            syn_df = self.load_synthetic_data()
            
            qty_col = 'qty' if 'qty' in syn_df.columns else 'units_sold'
            self.validate_data(syn_df, ['product_id', 'price_charged', qty_col], "synthetic_data")
            
            logging.info("Building feature store...")
            fs_df = self.build_feature_store(pd.DataFrame(), syn_df)
            
            logging.info(f"Summary:")
            logging.info(f"Synthetic data shape: {syn_df.shape}")
            logging.info(f"Feature store shape: {fs_df.shape}")
            logging.info(f"Feature store nulls:\n{fs_df.isnull().sum()}")
            
            from models.elasticity import ElasticityModel
            elast = ElasticityModel()
            elast.fit(fs_df, 
                      category_col='category',
                      price_col='price_charged', 
                      qty_col='units_sold')
            elast.save(PROCESSED_DIR / 'elasticity_model.pkl')
            logging.info("Elasticity model fitted and saved.")
            
            return {
                "synthetic": syn_df,
                "feature_store": fs_df
            }
        except Exception as e:
            logging.error(f"ETL pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    etl = PriceIQETL()
    etl.run_full_pipeline()
