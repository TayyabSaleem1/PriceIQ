import logging
import pandas as pd
import numpy as np
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from pytrends.request import TrendReq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DemandForecaster:
    def __init__(self, product_id, use_xgboost_residuals=True):
        self.product_id = product_id
        self.use_xgboost_residuals = use_xgboost_residuals
        self.prophet_model = None
        self.xgb_model = None

    def prepare_prophet_df(self, sales_df):
        df = sales_df[sales_df['product_id'] == self.product_id].copy()
        df = df.rename(columns={'date': 'ds', 'units_sold': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure regressors exist
        if 'promo_flag' not in df.columns:
            df['promo_flag'] = 0
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['ds'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
            
        return df[['ds', 'y', 'promo_flag', 'is_weekend']]

    def fetch_trends_signal(self, keyword, geo="US", timeframe="today 12-m"):
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
            trends_df = pytrends.interest_over_time()
            if not trends_df.empty and keyword in trends_df.columns:
                return trends_df[keyword]
        except Exception as e:
            logging.warning(f"Google Trends API failed for {keyword}: {str(e)}. Using fallback 50.0")
            
        # Fallback
        dates = pd.date_range(end=pd.Timestamp.today(), periods=52, freq='W')
        return pd.Series(data=50.0, index=dates)

    def fit(self, sales_df, trends_keyword=None):
        df = self.prepare_prophet_df(sales_df)
        
        if not PROPHET_AVAILABLE:
            # Fallback: use simple 7-day rolling mean as forecast
            self.prophet_model = None
            self._fallback_mean = sales_df['units_sold'].rolling(7).mean().iloc[-1]
            self._fallback_std = sales_df['units_sold'].std()
            return self
        
        self.prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        self.prophet_model.add_regressor('promo_flag')
        self.prophet_model.add_regressor('is_weekend')
        
        if trends_keyword:
            trend_series = self.fetch_trends_signal(trends_keyword)
            # Match trends to daily data (forward fill)
            trend_df = pd.DataFrame({'ds': trend_series.index, 'trend': trend_series.values})
            trend_df['ds'] = pd.to_datetime(trend_df['ds'])
            df = pd.merge_asof(df.sort_values('ds'), trend_df.sort_values('ds'), on='ds', direction='backward')
            df['trend'] = df['trend'].fillna(50.0)
            self.prophet_model.add_regressor('trend')
        else:
            self.has_trend = False
            
        self.prophet_model.fit(df)
        
        if self.use_xgboost_residuals:
            # Predict on train set to get residuals
            train_preds = self.prophet_model.predict(df)
            df['yhat'] = train_preds['yhat']
            df['residual'] = df['y'] - df['yhat']
            
            # Create features for XGBoost
            df['day_of_week'] = df['ds'].dt.dayofweek
            df['month'] = df['ds'].dt.month
            df['lag_7'] = df['y'].shift(7).fillna(method='bfill')
            df['lag_14'] = df['y'].shift(14).fillna(method='bfill')
            df['rolling_mean_7'] = df['y'].rolling(7).mean().fillna(method='bfill')
            
            features = ['day_of_week', 'month', 'lag_7', 'lag_14', 'promo_flag', 'rolling_mean_7']
            
            self.xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            self.xgb_model.fit(df[features], df['residual'])
            
            # Store last known values for future feature generation
            self.last_y = df['y'].values
            self.last_ds = df['ds'].max()

    def predict(self, horizon_days=7):
        if not self.prophet_model:
            dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon_days, freq='D')
            forecast_val = self._fallback_mean if hasattr(self, '_fallback_mean') else 10.0
            std_val = self._fallback_std if hasattr(self, '_fallback_std') else 2.0
            return pd.DataFrame({
                'ds': dates,
                'prophet_forecast': forecast_val,
                'xgb_adjustment': 0.0,
                'final_forecast': forecast_val,
                'lower_bound': max(0, forecast_val - std_val),
                'upper_bound': forecast_val + std_val
            })
            
        future = self.prophet_model.make_future_dataframe(periods=horizon_days)
        future['is_weekend'] = future['ds'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        future['promo_flag'] = 0  # Assume no promo in future by default, or could be parameterized
        
        if hasattr(self.prophet_model, 'extra_regressors') and 'trend' in self.prophet_model.extra_regressors:
            future['trend'] = 50.0 # simplified: assume flat trend in future if not available
            
        forecast = self.prophet_model.predict(future)
        
        result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon_days).copy()
        result_df = result_df.rename(columns={'yhat': 'prophet_forecast', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'})
        
        if self.use_xgboost_residuals and self.xgb_model:
            # Reconstruct features for future horizon
            future_features = pd.DataFrame({'ds': result_df['ds']})
            future_features['day_of_week'] = future_features['ds'].dt.dayofweek
            future_features['month'] = future_features['ds'].dt.month
            future_features['promo_flag'] = future['promo_flag'].tail(horizon_days).values
            
            # Use last known y values for lags
            if len(self.last_y) >= 14:
                lag_7 = list(self.last_y[-7:])
                lag_14 = list(self.last_y[-14:-7])
                rolling_7 = list(self.last_y[-7:])
            else:
                lag_7 = [0]*horizon_days
                lag_14 = [0]*horizon_days
                rolling_7 = [0]*horizon_days
                
            # Autoregressive generation for XGBoost features
            xgb_adj = []
            for i in range(horizon_days):
                row = future_features.iloc[i:i+1].copy()
                row['lag_7'] = lag_7[i] if i < len(lag_7) else lag_7[-1]
                row['lag_14'] = lag_14[i] if i < len(lag_14) else lag_14[-1]
                row['rolling_mean_7'] = np.mean(rolling_7[-7:]) if rolling_7 else 0
                
                adj = self.xgb_model.predict(row[['day_of_week', 'month', 'lag_7', 'lag_14', 'promo_flag', 'rolling_mean_7']])[0]
                xgb_adj.append(adj)
                
                # Update lists for next step assuming predicted value = prophet + adj
                pred_val = result_df.iloc[i]['prophet_forecast'] + adj
                lag_7.append(pred_val)
                lag_14.append(lag_7[i])
                rolling_7.append(pred_val)
                
            result_df['xgb_adjustment'] = xgb_adj
            result_df['final_forecast'] = result_df['prophet_forecast'] + result_df['xgb_adjustment']
        else:
            result_df['xgb_adjustment'] = 0.0
            result_df['final_forecast'] = result_df['prophet_forecast']
            
        return result_df[['ds', 'prophet_forecast', 'xgb_adjustment', 'final_forecast', 'lower_bound', 'upper_bound']]

    def forecast_accuracy(self, test_df):
        # Filter test_df for the product
        test_data = test_df[test_df['product_id'] == self.product_id].copy()
        if test_data.empty:
            raise ValueError("test_df contains no data for this product_id")
            
        test_data['ds'] = pd.to_datetime(test_data['date'])
        horizon = len(test_data)
        
        preds = self.predict(horizon_days=horizon)
        
        # Merge on ds to ensure alignment
        merged = pd.merge(test_data, preds, on='ds', how='inner')
        
        y_true = merged['units_sold'].values
        y_pred = merged['final_forecast'].values
        
        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
