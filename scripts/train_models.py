import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ==========================================
# PART 1: DEMAND FORECASTING (PROPHET)
# ==========================================
print("--- 1. Training Demand Forecasting Model (Prophet) ---")

# Load and prepare daily sales
daily = pd.read_csv('data/processed/daily_sales.csv', parse_dates=['date'])
daily = daily[['date', 'orders']].rename(columns={'date': 'ds', 'orders': 'y'})
daily = daily.sort_values('ds').reset_index(drop=True)

# Hold out last 30 days for evaluation
train_prophet = daily.iloc[:-30]
test_prophet = daily.iloc[-30:]

# Fit Prophet model
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
prophet_model.fit(train_prophet)

# Predict on test horizon
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='D')
forecast = prophet_model.predict(future)
forecast_test = forecast[['ds', 'yhat']].merge(test_prophet, on='ds', how='right')

# Metrics
prophet_mae = mean_absolute_error(test_prophet['y'], forecast_test['yhat'])
prophet_mape = mean_absolute_percentage_error(test_prophet['y'], forecast_test['yhat']) * 100

print(f"Prophet Demand Forecast Test MAE: {prophet_mae:.2f} orders")
print(f"Prophet Demand Forecast Test MAPE: {prophet_mape:.2f}%")

# Save Forecast Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_prophet['ds'], train_prophet['y'], label='Train (Actual)', color='blue')
ax.plot(test_prophet['ds'], test_prophet['y'], label='Test (Actual)', color='green')
ax.plot(forecast_test['ds'], forecast_test['yhat'], label='Forecast', color='red')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.15)
plt.title('Daily Demand Forecast (Prophet)')
plt.xlabel('Date')
plt.ylabel('Daily Orders')
plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/demand_forecast_prophet.png', dpi=150)
plt.close()

# Save Model
joblib.dump(prophet_model, 'models/prophet_demand_model.pkl')
print("Saved model: models/prophet_demand_model.pkl\n")

# ==========================================
# PART 2: DELIVERY TIME PREDICTION (XGBOOST)
# ==========================================
print("--- 2. Training Delivery Time Prediction Model (XGBoost) ---")

orders = pd.read_csv(
    'data/processed/orders_processed.csv',
    parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
)
items = pd.read_csv('data/processed/order_items_processed.csv')

# Aggregate item data per order
order_agg = items.groupby('order_id').agg(
    num_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    avg_seller_to_customer_km=('seller_to_customer_km', 'mean'),
    max_seller_to_customer_km=('seller_to_customer_km', 'max'),
    distinct_sellers=('seller_id', 'nunique'),
    distinct_products=('product_id', 'nunique')
).reset_index()

# Merge with orders dataset
df = orders.merge(order_agg, on='order_id', how='inner')
df = df.dropna(subset=['delivery_time_days']).copy()
df = df[df['delivery_time_days'] > 0].copy()

# Feature engineering
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_year'] = df['order_purchase_timestamp'].dt.year

# Label encode state
le_state = LabelEncoder()
df['customer_state_encoded'] = le_state.fit_transform(df['customer_state'])

feature_cols = [
    'num_items',
    'total_price',
    'total_freight',
    'avg_seller_to_customer_km',
    'max_seller_to_customer_km',
    'distinct_sellers',
    'distinct_products',
    'purchase_hour',
    'purchase_dayofweek',
    'purchase_month',
    'purchase_year',
    'customer_state_encoded'
]

X = df[feature_cols]
y = df['delivery_time_days']

# Time-based split (80% train, 20% test)
df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae_xgb = mean_absolute_error(y_test, y_pred)

print(f"XGBoost Test RMSE: {rmse:.2f} days")
print(f"XGBoost Test MAE: {mae_xgb:.2f} days")
print(f"XGBoost Test R² Score: {r2:.2f}")

# Feature Importance
importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop Predictive Features for Delivery Time:")
print(importance.head(5))

# Save XGBoost Artifacts
joblib.dump(xgb_model, 'models/xgb_delivery_time_model.pkl')
joblib.dump(le_state, 'models/state_label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("\nSaved artifacts to models/ directory successfully.")

# ==========================================
# PART 3: ANOMALY DETECTION (ISOLATION FOREST)
# ==========================================
print("\n--- 3. Training Delivery Anomaly Detection (Isolation Forest) ---")
from sklearn.ensemble import IsolationForest

anomaly_features = df[['delivery_time_days', 'total_freight', 'avg_seller_to_customer_km', 'num_items']]

iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['is_anomaly'] = iso_forest.fit_predict(anomaly_features)

anomalies = df[df['is_anomaly'] == -1]
print(f"Detected {len(anomalies)} anomalous deliveries out of {len(df)} total orders (~2%).")

joblib.dump(iso_forest, 'models/isolation_forest_anomaly.pkl')
print("Saved model: models/isolation_forest_anomaly.pkl")