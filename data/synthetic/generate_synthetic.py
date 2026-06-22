import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SYNTHETIC_DIR, PRODUCT_CATEGORIES

def generate_inventory(num_products=50):
    np.random.seed(42)
    
    product_ids = [f"P{i:03d}" for i in range(1, num_products + 1)]
    categories = np.random.choice(PRODUCT_CATEGORIES, size=num_products)
    
    # Generate realistic names
    name_prefixes = {"electronics": ["Smart", "Pro", "Ultra", "Max"], 
                     "clothing": ["Cotton", "Casual", "Formal", "Sport"],
                     "home_garden": ["Eco", "Modern", "Classic", "Premium"],
                     "sports": ["Active", "Fit", "Pro", "Elite"],
                     "beauty": ["Natural", "Glow", "Pure", "Luxe"]}
    name_suffixes = {"electronics": ["TV", "Headphones", "Speaker", "Monitor", "Phone"],
                     "clothing": ["Shirt", "Pants", "Jacket", "Dress", "Socks"],
                     "home_garden": ["Chair", "Table", "Lamp", "Rug", "Vase"],
                     "sports": ["Mat", "Dumbbell", "Bottle", "Band", "Tracker"],
                     "beauty": ["Cream", "Serum", "Lotion", "Mask", "Oil"]}
    
    product_names = []
    for cat in categories:
        prefix = np.random.choice(name_prefixes[cat])
        suffix = np.random.choice(name_suffixes[cat])
        product_names.append(f"{prefix} {suffix}")
        
    cost_prices = np.round(np.random.uniform(5.0, 500.0, size=num_products), 2)
    current_stocks = np.random.randint(0, 501, size=num_products)
    reorder_points = (np.max(current_stocks) * 0.2).astype(int) * np.ones(num_products, dtype=int)
    
    # random created_at dates in 2022
    created_dates = pd.date_range(start="2022-01-01", end="2022-12-31").to_series().sample(num_products, replace=True, random_state=42).dt.strftime('%Y-%m-%d').values
    
    inventory_df = pd.DataFrame({
        "product_id": product_ids,
        "product_name": product_names,
        "category": categories,
        "cost_price": cost_prices,
        "current_stock": current_stocks,
        "reorder_point": reorder_points,
        "created_at": created_dates
    })
    return inventory_df

def generate_sales_history(inventory_df, start_date="2023-01-01", end_date="2024-12-31"):
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, end=end_date)
    product_ids = inventory_df["product_id"].values
    cost_prices = inventory_df.set_index("product_id")["cost_price"].to_dict()
    
    sales_data = []
    
    for pid in product_ids:
        cost = cost_prices[pid]
        base_price = cost * 1.4
        
        # Monthly seasonality pattern
        t = np.arange(len(dates))
        monthly_seasonality = 10 * np.sin(2 * np.pi * t / 365)
        
        # Promotions
        promo_flags = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
        
        # Base units
        base_units = np.random.poisson(20, size=len(dates))
        
        for i, date in enumerate(dates):
            # Weekend bump
            weekend_bump = 1.2 if date.weekday() >= 5 else 1.0
            
            # Promo bump
            promo_bump = 1.3 if promo_flags[i] == 1 else 1.0
            
            # Final units sold
            units_sold = max(0, int((base_units[i] + monthly_seasonality[i]) * weekend_bump * promo_bump))
            
            # Price charged with noise
            price_charged = np.round(base_price * np.random.uniform(0.85, 1.15), 2)
            
            # Competitor price with noise
            comp_price = np.round(price_charged * np.random.uniform(0.9, 1.1), 2)
            
            sales_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "units_sold": units_sold,
                "price_charged": price_charged,
                "competitor_price": comp_price,
                "promo_flag": promo_flags[i]
            })
            
    return pd.DataFrame(sales_data)

if __name__ == "__main__":
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    
    print("Generating synthetic inventory...")
    inventory_df = generate_inventory()
    inventory_path = SYNTHETIC_DIR / "inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    
    print("Generating synthetic sales history...")
    sales_df = generate_sales_history(inventory_df)
    sales_path = SYNTHETIC_DIR / "sales_history.csv"
    sales_df.to_csv(sales_path, index=False)
    
    print(f"Generated {len(inventory_df)} rows in {inventory_path}")
    print(f"Generated {len(sales_df)} rows in {sales_path}")
    print("Done!")
