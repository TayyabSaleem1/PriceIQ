import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_MODEL, PROCESSED_DIR

class SentimentAnalyzer:
    def __init__(self, model_name=SENTIMENT_MODEL):
        self.pipeline = pipeline("sentiment-analysis", model=model_name, device=-1, truncation=True, max_length=512)
        print("Sentiment model loaded")

    def analyze_batch(self, texts: list, batch_size=32):
        if not texts:
            return []
            
        # Handle empty or nan texts
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        results = []
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i+batch_size]
            # pipeline handles batching automatically when passed a list
            batch_results = self.pipeline(batch)
            for text, res in zip(batch, batch_results):
                results.append({
                    'text': text,
                    'label': res['label'],
                    'score': res['score']
                })
        return results

    def analyze_reviews_df(self, reviews_df):
        df = reviews_df.copy()
        
        if 'Text' not in df.columns:
            raise ValueError("DataFrame must contain 'Text' column")
            
        results = self.analyze_batch(df['Text'].tolist())
        
        df['predicted_sentiment'] = [r['label'] for r in results]
        df['confidence'] = [r['score'] for r in results]
        
        return df

    def competitor_weakness_score(self, reviews_df, competitor_col="competitor_name"):
        if competitor_col not in reviews_df.columns:
            raise ValueError(f"DataFrame must contain '{competitor_col}' column")
            
        scores = {}
        for comp, group in reviews_df.groupby(competitor_col):
            if len(group) == 0:
                continue
                
            neg_reviews = group[group['predicted_sentiment'] == 'NEGATIVE']
            neg_fraction = len(neg_reviews) / len(group)
            
            if len(neg_reviews) > 0:
                mean_conf = neg_reviews['confidence'].mean()
            else:
                mean_conf = 0.0
                
            scores[comp] = neg_fraction * mean_conf
            
        return scores

    def aspect_summary(self, reviews_df, aspects=None):
        if aspects is None:
            aspects = ["quality", "delivery", "price", "service"]
            
        summary = {}
        for aspect in aspects:
            # Case-insensitive search for aspect keyword
            mask = reviews_df['Text'].str.contains(aspect, case=False, na=False)
            aspect_df = reviews_df[mask]
            
            if len(aspect_df) == 0:
                summary[aspect] = {
                    'positive_count': 0,
                    'negative_count': 0,
                    'sentiment_ratio': 0.0
                }
                continue
                
            pos_count = len(aspect_df[aspect_df['predicted_sentiment'] == 'POSITIVE'])
            neg_count = len(aspect_df[aspect_df['predicted_sentiment'] == 'NEGATIVE'])
            
            total = pos_count + neg_count
            ratio = pos_count / total if total > 0 else 0.0
            
            summary[aspect] = {
                'positive_count': pos_count,
                'negative_count': neg_count,
                'sentiment_ratio': ratio
            }
            
        return summary

    def save_cache(self, df, filepath=None):
        if filepath is None:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            filepath = PROCESSED_DIR / "sentiment_cache.pkl"
        joblib.dump(df, filepath)

    def load_cache(self, filepath=None):
        if filepath is None:
            filepath = PROCESSED_DIR / "sentiment_cache.pkl"
        return joblib.load(filepath)
