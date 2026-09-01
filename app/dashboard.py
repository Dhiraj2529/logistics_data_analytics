import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Logistics & Supply Chain Analytics",
    page_icon="🚚",
    layout="wide"
)

# Load datasets
@st.cache_data
def load_data():
    orders = pd.read_csv('data/processed/orders_processed.csv', parse_dates=['order_purchase_timestamp'])
    items = pd.read_csv('data/processed/order_items_processed.csv')
    daily_sales = pd.read_csv('data/processed/daily_sales.csv', parse_dates=['date'])
    
    # Load inventory summary
    if os.path.exists('data/processed/inventory_optimization_summary.csv'):
        inventory = pd.read_csv('data/processed/inventory_optimization_summary.csv')
    else:
        inventory = pd.DataFrame()
        
    return orders, items, daily_sales, inventory

# Load trained models
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_delivery_time_model.pkl')
    le_state = joblib.load('models/state_label_encoder.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    prophet_model = joblib.load('models/prophet_demand_model.pkl')
    return xgb_model, le_state, feature_cols, prophet_model

orders, items, daily_sales, inventory = load_data()
xgb_model, le_state, feature_cols, prophet_model = load_models()

# Navigation Sidebar
st.sidebar.title("🚚 Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["📊 Executive Overview", "📈 Demand Forecasting", "⏱️ Delivery Time Predictor", "📦 Inventory & Route Optimization"]
)

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.title("📊 Supply Chain & Logistics Performance Dashboard")
    st.markdown("Real-time logistics visibility, fulfillment metrics, and SLA tracking across Brazil.")

    # KPI Metrics
    orders['on_time'] = orders['delay_days'] <= 0
    otd_rate = orders['on_time'].mean() * 100
    avg_delivery = orders['delivery_time_days'].mean()
    avg_delay = orders['delay_days'].mean()
    
    order_freight = items.groupby('order_id')['freight_value'].sum().reset_index()
    avg_freight = order_freight['freight_value'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("On-Time Delivery Rate", f"{otd_rate:.1f}%", delta=f"{otd_rate - 90:.1f}% vs 90% SLA")
    col2.metric("Avg Delivery Lead Time", f"{avg_delivery:.1f} days")
    col3.metric("Avg Delay Metric", f"{avg_delay:.1f} days")
    col4.metric("Avg Freight per Order", f"R$ {avg_freight:.2f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Daily Order Volume Trend")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['orders'], name="Orders", line=dict(color="#1f77b4")))
        fig_ts.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title="Daily Orders")
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        st.subheader("On-Time Delivery Rate by State")
        state_perf = orders.groupby('customer_state').agg(
            total=('order_id', 'count'),
            on_time=('on_time', 'sum')
        ).reset_index()
        state_perf['otd_pct'] = (state_perf['on_time'] / state_perf['total']) * 100
        state_perf = state_perf.sort_values('otd_pct', ascending=False)
        
        fig_state = px.bar(
            state_perf, 
            x='customer_state', 
            y='otd_pct', 
            labels={'customer_state': 'State', 'otd_pct': 'OTD Rate (%)'},
            color='otd_pct',
            color_continuous_scale='RdYlGn'
        )
        fig_state.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), yaxis_range=[0, 100])
        st.plotly_chart(fig_state, use_container_width=True)

# ==========================================
# PAGE 2: DEMAND FORECASTING (PROPHET)
# ==========================================
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting with Prophet")
    st.markdown("Project future daily order volumes with uncertainty intervals for proactive capacity planning.")

    forecast_days = st.slider("Forecast Horizon (Days ahead)", min_value=7, max_value=60, value=30, step=7)

    if st.button("Generate Forecast"):
        future = prophet_model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = prophet_model.predict(future)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=daily_sales['date'], y=daily_sales['orders'],
            name="Actual Orders", mode="lines", line=dict(color="#1f77b4")
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            name="Predicted Demand", mode="lines", line=dict(color="orange")
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(255,165,0,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Uncertainty Range (80%)"
        ))
        fig_forecast.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Daily Orders",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        recent_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        recent_forecast.columns = ['Date', 'Projected Orders', 'Lower Bound', 'Upper Bound']
        st.subheader("Forecasted Quantities")
        st.dataframe(recent_forecast.reset_index(drop=True), use_container_width=True)

# ==========================================
# PAGE 3: DELIVERY TIME PREDICTOR (XGBOOST)
# ==========================================
elif page == "⏱️ Delivery Time Predictor":
    st.title("⏱️ Real-Time Delivery Duration Predictor")
    st.markdown("Predict expected shipping duration (in days) using machine learning before dispatch.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            state = st.selectbox("Destination State", sorted(le_state.classes_))
            num_items = st.number_input("Number of Items in Order", min_value=1, max_value=20, value=1)
            total_price = st.number_input("Total Order Price (R$)", min_value=5.0, max_value=5000.0, value=120.0, step=10.0)
            total_freight = st.number_input("Total Freight Value (R$)", min_value=1.0, max_value=500.0, value=25.0, step=5.0)

        with col2:
            distance_km = st.slider("Estimated Distance (km)", min_value=5.0, max_value=3500.0, value=450.0, step=25.0)
            distinct_sellers = st.selectbox("Distinct Sellers Count", [1, 2, 3, 4], index=0)
            purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=datetime.datetime.now().month - 1)
            purchase_dayofweek = st.selectbox("Day of Week (0=Monday, 6=Sunday)", list(range(7)), index=0)

        submit = st.form_submit_button("⚡ Predict Delivery Lead Time")

    if submit:
        state_encoded = le_state.transform([state])[0]
        input_data = pd.DataFrame([{
            'num_items': num_items,
            'total_price': total_price,
            'total_freight': total_freight,
            'avg_seller_to_customer_km': distance_km,
            'max_seller_to_customer_km': distance_km,
            'distinct_sellers': distinct_sellers,
            'distinct_products': 1,
            'purchase_hour': 14,
            'purchase_dayofweek': purchase_dayofweek,
            'purchase_month': purchase_month,
            'purchase_year': 2018,
            'customer_state_encoded': state_encoded
        }])

        pred_days = xgb_model.predict(input_data[feature_cols])[0]

        st.success(f"📦 **Estimated Delivery Lead Time:** `{pred_days:.1f} days`")
        if pred_days > 20:
            st.warning("⚠️ High lead-time risk detected. Regional transit delay buffer recommended.")
        else:
            st.info("✅ Lead time is within standard transit operating benchmarks.")

# ==========================================
# PAGE 4: INVENTORY & ROUTE OPTIMIZATION
# ==========================================
elif page == "📦 Inventory & Route Optimization":
    st.title("📦 Inventory Control & Routing Optimization")
    
    st.subheader("1. Inventory Reorder Points (ROP) & Safety Stock")
    st.markdown("Category-level safety stock calculated for a 95% service level ($Z=1.65$) with 5-day supplier lead time.")
    
    if not inventory.empty:
        st.dataframe(inventory.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("2. Regional Fulfillment Clusters & Route Optimization")
    st.markdown("Customer delivery locations grouped into sub-hubs, reducing route transit distances by **~45.8%** via nearest-neighbor TSP heuristics.")
    
    if os.path.exists('reports/figures/sp_delivery_clusters.png'):
        st.image('reports/figures/sp_delivery_clusters.png', caption="São Paulo Delivery Zone Optimization (K-Means Clustering)")   