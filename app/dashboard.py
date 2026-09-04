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
    page_icon="📦",
    layout="wide"
)

# Custom Enterprise CSS (Clean layout, no glows, high readability)
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dcdfe6;
        padding: 15px;
        border-radius: 6px;
        box-shadow: none !important;
    }
    div[data-testid="stMetric"] label { color: #586069 !important; font-weight: 600; font-size: 13px; }
    div[data-testid="stDataFrame"] { border: 1px solid #dcdfe6; border-radius: 6px; background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #dcdfe6; }
    </style>
""", unsafe_allow_html=True)

# Safe Data Loading with Fallbacks
@st.cache_data
def load_data():
    try:
        orders = pd.read_csv('data/processed/orders_processed.csv', parse_dates=['order_purchase_timestamp'])
    except Exception:
        orders = pd.DataFrame()
        
    try:
        items = pd.read_csv('data/processed/order_items_processed.csv')
    except Exception:
        items = pd.DataFrame()
        
    try:
        daily_sales = pd.read_csv('data/processed/daily_sales.csv', parse_dates=['date'])
    except Exception:
        daily_sales = pd.DataFrame()
        
    if os.path.exists('data/processed/inventory_optimization_summary.csv'):
        inventory = pd.read_csv('data/processed/inventory_optimization_summary.csv')
    else:
        inventory = pd.DataFrame()
        
    return orders, items, daily_sales, inventory

@st.cache_resource
def load_models():
    models = {}
    try:
        models['xgb'] = joblib.load('models/xgb_delivery_time_model.pkl')
        models['le_state'] = joblib.load('models/state_label_encoder.pkl')
        models['features'] = joblib.load('models/feature_columns.pkl')
        models['prophet'] = joblib.load('models/prophet_demand_model.pkl')
    except Exception as e:
        models['error'] = str(e)
    return models

orders, items, daily_sales, inventory = load_data()
models = load_models()

# ==========================================
# SIDEBAR DATA LOADER & UPLOADER
# ==========================================
st.sidebar.title("📦 Logistics Hub")
page = st.sidebar.radio(
    "Select Module",
    ["📊 Executive Overview", "📈 Demand Forecasting", "⏱️ Delivery Time Predictor", "📦 Inventory & Route Optimization"]
)

# ==========================================
# INTERACTIVE USER GUIDE (SIDEBAR)
# ==========================================
with st.sidebar.expander("📖 User Guide & Instructions", expanded=False):
    st.markdown("""
    **Welcome to the Logistics Hub!** 
    Follow these quick steps to explore the platform:
    
    1. **📊 Executive Overview:**
       - Organized across clean tabs (**Volume & Time Trends**, **Regional & SLA Performance**, and **Deep Analytics & Diagnostics**).
       - View high-level metrics (OTD Rate, Lead Times, Freight costs).
       - Analyze order volume trends, delivery time distributions, correlation matrices, distance scatter plots, state OTD rates, delay time-series, day-hour heatmaps, anomaly alerts, and CSV exports.
       
    2. **📈 Demand Forecasting:**
       - Use the slider to select a forecast horizon (7 to 60 days).
       - Click **Generate Forecast** to project future order demand with 80% confidence bands.
       
    3. **⏱️ Delivery Time Predictor:**
       - Input order parameters (Destination State, Items, Price, Distance).
       - Click **Predict Delivery Lead Time** to get real-time ML duration estimates.
       
    4. **📦 Inventory & Route Optimization:**
       - Review category-level Safety Stock and Reorder Points (ROP).
       - Inspect regional cluster maps for delivery route efficiency.
       
    5. **📂 Custom Data Testing:**
       - Toggle to **Upload Custom CSV** in the sidebar data mode to test your own data files live!
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Source Configuration")
data_mode = st.sidebar.radio("Choose Data Mode", ["Default Dataset", "Upload Custom CSV"])

# Initialize empty DataFrames
orders, items, daily_sales, inventory = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if data_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload orders CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Load custom data
            orders = pd.read_csv(uploaded_file, parse_dates=['order_purchase_timestamp'])
            items = orders.copy() # fallback mapping if items file isn't separate
            
            # Auto-generate daily sales if date column exists
            if 'order_purchase_timestamp' in orders.columns:
                orders['date'] = orders['order_purchase_timestamp'].dt.date
                daily_sales = orders.groupby('date').size().reset_index(name='orders')
                
            st.sidebar.success("✅ Custom dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading custom file: {e}")
    else:
        st.sidebar.warning("⚠️ Please upload a CSV file to render analytics.")
else:
    # Default Safe Data Loading with Fallbacks
    @st.cache_data
    def load_default_data():
        try:
            o = pd.read_csv('data/processed/orders_processed.csv', parse_dates=['order_purchase_timestamp'])
        except Exception:
            o = pd.DataFrame()
        try:
            i = pd.read_csv('data/processed/order_items_processed.csv')
        except Exception:
            i = pd.DataFrame()
        try:
            ds = pd.read_csv('data/processed/daily_sales.csv', parse_dates=['date'])
        except Exception:
            ds = pd.DataFrame()
        if os.path.exists('data/processed/inventory_optimization_summary.csv'):
            inv = pd.read_csv('data/processed/inventory_optimization_summary.csv')
        else:
            inv = pd.DataFrame()
        return o, i, ds, inv

    orders, items, daily_sales, inventory = load_default_data()

st.sidebar.markdown("---")

# ==========================================
# SIDEBAR FILTERS (Interactive Analysis)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Global Filters")

# Check if orders data is available to extract states/dates
if not orders.empty and 'customer_state' in orders.columns:
    # 1. State Filter
    all_states = sorted(orders['customer_state'].dropna().unique())
    selected_states = st.sidebar.multiselect(
        "Filter by Destination State(s)",
        options=all_states,
        default=all_states[:3], # Select first 3 by default to keep it fast
        help="Select specific Brazilian states to filter the metrics, charts, and predictions."
    )
    
    # Apply State Filter to DataFrame
    if selected_states:
        filtered_orders = orders[orders['customer_state'].isin(selected_states)]
    else:
        filtered_orders = orders.copy()
else:
    filtered_orders = orders.copy()

# 2. Date Range Filter (if purchase timestamp exists)
if not filtered_orders.empty and 'order_purchase_timestamp' in filtered_orders.columns:
    min_date = filtered_orders['order_purchase_timestamp'].min().date()
    max_date = filtered_orders['order_purchase_timestamp'].max().date()
    
    selected_date_range = st.sidebar.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select a start and end date range to narrow down order telemetry."
    )
    
    # Apply Date Range Filter if both start and end dates are selected
    if len(selected_date_range) == 2:
        start_d, end_d = selected_date_range
        filtered_orders = filtered_orders[
            (filtered_orders['order_purchase_timestamp'].dt.date >= start_d) & 
            (filtered_orders['order_purchase_timestamp'].dt.date <= end_d)
        ]

st.sidebar.success(f"📊 Active Records: {len(filtered_orders):,} orders loaded.")
st.sidebar.markdown("---")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW (TABBED & CLUSTERED)
# ==========================================
if page == "📊 Executive Overview":
    st.title("Supply Chain & Logistics Performance Dashboard")
    st.markdown("Operational overview of fulfillment metrics, SLAs, and delivery timelines across Brazil.")

    if filtered_orders.empty or items.empty:
        st.error("Processed data files not found or filtered dataset is empty. Please check your filters or data source.")
    else:
        # Use filtered_orders for calculations so sidebar filters work instantly!
        filtered_orders['on_time'] = filtered_orders['delay_days'] <= 0
        otd_rate = filtered_orders['on_time'].mean() * 100
        avg_delivery = filtered_orders['delivery_time_days'].mean()
        avg_delay = filtered_orders['delay_days'].mean()
        
        # Match items with filtered orders
        filtered_items = items[items['order_id'].isin(filtered_orders['order_id'])]
        order_freight = filtered_items.groupby('order_id')['freight_value'].sum().reset_index()
        avg_freight = order_freight['freight_value'].mean() if not order_freight.empty else 0.0

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("On-Time Delivery Rate", f"{otd_rate:.1f}%", delta=f"{otd_rate - 90:.1f}% vs SLA", help="Percentage of orders delivered on or before promised SLA.")
        col2.metric("Avg Delivery Lead Time", f"{avg_delivery:.1f} days", help="Average total duration from purchase to doorstep delivery.")
        col3.metric("Avg Delay Metric", f"{avg_delay:.1f} days", help="Average variance between actual delivery date and estimated SLA.")
        col4.metric("Avg Freight per Order", f"R$ {avg_freight:.2f}", help="Average shipping cost per processed order.")

        st.markdown("###")

        # =========================================================================
        # TABBED ORGANIZATION FOR ALL 9+ VISUALIZATIONS
        # =========================================================================
        tab_vol, tab_reg, tab_adv = st.tabs([
            "📈 Volume & Time Trends", 
            "🗺️ Regional & SLA Performance", 
            "🔬 Deep Analytics & Diagnostics"
        ])

        # --- TAB 1: VOLUME & TIME TRENDS ---
        with tab_vol:
            st.info("💡 **Feature Tale:** Tracks macro-level order volume velocity, longitudinal delay trends, and day/hour operational density patterns.")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("Daily Order Volume Trend")
                filtered_daily_sales = filtered_orders.groupby(filtered_orders['order_purchase_timestamp'].dt.date).size().reset_index(name='orders')
                filtered_daily_sales.columns = ['date', 'orders']
                
                if not filtered_daily_sales.empty:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=filtered_daily_sales['date'], y=filtered_daily_sales['orders'], 
                        name="Orders", line=dict(color="#2b5c8f", width=2)
                    ))
                    fig_ts.update_layout(
                        height=350, margin=dict(l=10, r=10, t=20, b=10), 
                        xaxis_title="Date", yaxis_title="Daily Orders",
                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                    )
                    fig_ts.update_xaxes(showgrid=True, gridcolor="#f1f3f5")
                    fig_ts.update_yaxes(showgrid=True, gridcolor="#f1f3f5")
                    st.plotly_chart(fig_ts, use_container_width=True)

            with col_t2:
                st.subheader("⏳ Average Delay Over Time")
                if not filtered_orders.empty and 'order_purchase_timestamp' in filtered_orders.columns and 'delay_days' in filtered_orders.columns:
                    df_delay_temp = filtered_orders.copy()
                    df_delay_temp['order_purchase_timestamp'] = pd.to_datetime(df_delay_temp['order_purchase_timestamp'])
                    daily_delay = df_delay_temp.set_index('order_purchase_timestamp').resample('D')['delay_days'].mean().reset_index()
                    
                    fig_delay = px.line(
                        daily_delay, x='order_purchase_timestamp', y='delay_days',
                        title="Daily Average Delay (days)",
                        labels={'order_purchase_timestamp': 'Purchase Date', 'delay_days': 'Average Delay (Days)'}
                    )
                    fig_delay.update_layout(
                        height=350, margin=dict(l=10, r=10, t=20, b=10),
                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                    )
                    fig_delay.update_xaxes(showgrid=True, gridcolor="#f1f3f5")
                    fig_delay.update_yaxes(showgrid=True, gridcolor="#f1f3f5")
                    st.plotly_chart(fig_delay, use_container_width=True)

            st.markdown("###")
            st.subheader("🕒 Order Volume by Day and Hour")
            if not filtered_orders.empty and 'order_purchase_timestamp' in filtered_orders.columns:
                df_heat_temp = filtered_orders.copy()
                df_heat_temp['hour'] = pd.to_datetime(df_heat_temp['order_purchase_timestamp']).dt.hour
                df_heat_temp['day_of_week'] = pd.to_datetime(df_heat_temp['order_purchase_timestamp']).dt.day_name()
                
                heat_data = df_heat_temp.pivot_table(index='day_of_week', columns='hour', values='order_id', aggfunc='count').fillna(0)
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heat_data = heat_data.reindex(days_order).dropna(how='all')
                
                if not heat_data.empty:
                    fig_heat = px.imshow(
                        heat_data,
                        labels=dict(x="Hour of Day", y="Day of Week", color="Order Count"),
                        x=heat_data.columns,
                        y=heat_data.index,
                        title="Order Density Heatmap by Day of Week and Hour",
                        color_continuous_scale="Blues"
                    )
                    fig_heat.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

        # --- TAB 2: REGIONAL & SLA PERFORMANCE ---
        with tab_reg:
            st.info("💡 **Feature Tale:** Evaluates geographic compliance rates and shipping variance outliers across state destinations.")

            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.subheader("✅ On-Time Delivery Rate by State")
                state_otd = filtered_orders.groupby('customer_state').agg(
                    total_orders=('order_id', 'count'),
                    on_time=('on_time', 'sum')
                ).reset_index()
                state_otd['otd_rate'] = (state_otd['on_time'] / state_otd['total_orders']) * 100
                state_otd = state_otd.sort_values('otd_rate', ascending=True)

                fig_otd = px.bar(
                    state_otd, x='customer_state', y='otd_rate',
                    title="On-Time Delivery Rate by State (%)",
                    labels={'otd_rate': 'OTD %', 'customer_state': 'Destination State'},
                    color='otd_rate', color_continuous_scale='blues'
                )
                fig_otd.update_layout(
                    height=400, margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", coloraxis_showscale=False
                )
                fig_otd.update_xaxes(showgrid=False)
                fig_otd.update_yaxes(showgrid=True, gridcolor="#f1f3f5", range=[0, 100])
                st.plotly_chart(fig_otd, use_container_width=True)

            with col_r2:
                st.subheader("📦 Delivery Time Distribution by State")
                fig_box = px.box(
                    filtered_orders,
                    x='customer_state', 
                    y='delivery_time_days',
                    title="Delivery Duration Spread & Outliers by Customer State",
                    labels={'delivery_time_days': 'Delivery Duration (Days)', 'customer_state': 'Destination State'},
                    color='customer_state'
                )
                fig_box.update_layout(
                    height=400, margin=dict(l=10, r=10, t=30, b=10), 
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", showlegend=False
                )
                fig_box.update_xaxes(showgrid=False)
                fig_box.update_yaxes(showgrid=True, gridcolor="#f1f3f5")
                st.plotly_chart(fig_box, use_container_width=True)

        # --- TAB 3: DEEP ANALYTICS & DIAGNOSTICS ---
        with tab_adv:
            st.info("💡 **Feature Tale:** Multi-variable correlation matrices, geometric distance scatter impact analysis, ML anomaly detection alerts, and filtered data exports.")

            order_agg = filtered_items.groupby('order_id').agg(
                total_price=('price', 'sum'),
                total_freight=('freight_value', 'sum'),
                avg_distance=('seller_to_customer_km', 'mean') if 'seller_to_customer_km' in filtered_items.columns else ('freight_value', 'count'),
                num_items=('order_item_id', 'count') if 'order_item_id' in filtered_items.columns else ('price', 'count')
            ).reset_index()

            corr_df = filtered_orders.merge(order_agg, on='order_id', how='left')

            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.subheader("🔗 Correlation Heatmap")
                target_corr_cols = ['delivery_time_days', 'delay_days', 'total_price', 'total_freight']
                if 'avg_distance' in corr_df.columns:
                    target_corr_cols.append('avg_distance')
                corr_subset = corr_df[target_corr_cols].dropna()

                if not corr_subset.empty:
                    fig_corr = px.imshow(
                        corr_subset.corr(), text_auto=".2f", aspect="auto",
                        title="Correlation Matrix of Logistics Variables",
                        color_continuous_scale="RdBu_r", range_color=[-1, 1]
                    )
                    fig_corr.update_layout(
                        height=400, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

            with col_a2:
                st.subheader("📈 Distance vs Delivery Time")
                if not corr_df.empty and 'avg_distance' in corr_df.columns and 'delivery_time_days' in corr_df.columns:
                    fig_scatter = px.scatter(
                        corr_df, x='avg_distance', y='delivery_time_days',
                        color='delay_days' if 'delay_days' in corr_df.columns else None, 
                        size='total_freight' if 'total_freight' in corr_df.columns else None,
                        hover_data=['order_id'] if 'order_id' in corr_df.columns else None,
                        title="Impact of Distance on Delivery Time and Cost",
                        labels={'avg_distance': 'Average Distance (km)', 'delivery_time_days': 'Delivery Duration (Days)'},
                        color_continuous_scale="Viridis"
                    )
                    fig_scatter.update_layout(
                        height=400, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            col_sub1, col_sub2 = st.columns(2)

            with col_sub1:
                st.subheader("⚠️ Anomaly Alerts")
                anomaly_file_path = 'data/processed/anomalies_detected.csv'
                if os.path.exists(anomaly_file_path):
                    anomalies_df = pd.read_csv(anomaly_file_path)
                    st.warning(f"🚨 Detected {len(anomalies_df):,} logistical anomalies flagged by ML models.")
                    st.dataframe(anomalies_df[['order_id', 'customer_state', 'delivery_time_days', 'delay_days']].head(5), use_container_width=True)
                else:
                    if not filtered_orders.empty and 'delivery_time_days' in filtered_orders.columns:
                        mean_dur = filtered_orders['delivery_time_days'].mean()
                        std_dur = filtered_orders['delivery_time_days'].std()
                        threshold = mean_dur + (3 * std_dur)
                        anomalies_df = filtered_orders[filtered_orders['delivery_time_days'] > threshold]
                        if not anomalies_df.empty:
                            st.warning(f"🚨 Flagged {len(anomalies_df):,} high-latency delivery outliers (> {threshold:.1f} days).")
                            display_cols = [c for c in ['order_id', 'customer_state', 'delivery_time_days', 'delay_days'] if c in anomalies_df.columns]
                            st.dataframe(anomalies_df[display_cols].head(5), use_container_width=True)
                        else:
                            st.success("✅ No extreme logistical delivery anomalies detected.")

            with col_sub2:
                st.subheader("📥 Download Filtered Data")
                st.markdown("Export your active filtered subset for external business reporting.")
                if not filtered_orders.empty:
                    csv_data = filtered_orders.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Filtered Orders CSV",
                        data=csv_data,
                        file_name='filtered_orders_export.csv',
                        mime='text/csv',
                        help="Click to download the current filtered dataset.",
                        use_container_width=True
                    )

# ==========================================
# PAGE 2: DEMAND FORECASTING (PROPHET)
# ==========================================
elif page == "📈 Demand Forecasting":
    st.title("Demand Forecasting with Prophet")
    st.markdown("Project future daily order volumes with confidence intervals.")

    if 'prophet' not in models:
        st.error("Prophet model not found. Please run `python scripts/train_models.py` first.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            forecast_days = st.slider("Select Forecast Horizon (Days ahead)", min_value=7, max_value=60, value=30, step=7)
        with col_ctrl2:
            st.write("###")
            generate_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

        if generate_btn or 'forecast_run' in st.session_state:
            with st.spinner("🔄 Running Prophet time-series forecasting model... Please wait."):
                st.session_state['forecast_run'] = True
                prophet_model = models['prophet']
                future = prophet_model.make_future_dataframe(periods=forecast_days, freq='D')
                forecast = prophet_model.predict(future)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=daily_sales['date'], y=daily_sales['orders'],
                name="Actual Orders", mode="lines", line=dict(color="#2b5c8f", width=1.5)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                name="Predicted Demand", mode="lines", line=dict(color="#d9534f", width=2)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(217,83,79,0.1)',
                line=dict(color='rgba(255,255,255,0)'), name="Uncertainty Range (80%)"
            ))
            fig_forecast.update_layout(
                height=450, xaxis_title="Date", yaxis_title="Daily Orders",
                hovermode="x unified", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
            )
            fig_forecast.update_xaxes(showgrid=True, gridcolor="#f1f3f5")
            fig_forecast.update_yaxes(showgrid=True, gridcolor="#f1f3f5")
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Forecasted Quantities Table")
            recent_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
            recent_forecast.columns = ['Date', 'Projected Orders', 'Lower Bound', 'Upper Bound']
            st.dataframe(recent_forecast.reset_index(drop=True), use_container_width=True, height=250)

# ==========================================
# PAGE 3: DELIVERY TIME PREDICTOR (XGBOOST)
# ==========================================
elif page == "⏱️ Delivery Time Predictor":
    st.title("Real-Time Delivery Duration Predictor")
    st.markdown("Estimate expected shipping duration in days prior to order dispatch using machine learning.")

    if 'xgb' not in models:
        st.error("XGBoost model not found. Please run `python scripts/train_models.py` first.")
    else:
        xgb_model = models['xgb']
        le_state = models['le_state']
        feature_cols = models['features']

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                state = st.selectbox("Destination State", sorted(le_state.classes_), help="Select the target state code in Brazil.")
                num_items = st.number_input("Number of Items in Order", min_value=1, max_value=20, value=1, help="Total physical items included in the order bundle.")
                total_price = st.number_input("Total Order Price (R$)", min_value=5.0, max_value=5000.0, value=120.0, step=10.0, help="Total monetary value of the goods.")
                total_freight = st.number_input("Total Freight Value (R$)", min_value=1.0, max_value=500.0, value=25.0, step=5.0, help="Total shipping cost.")

            with col2:
                distance_km = st.slider("Estimated Distance (km)", min_value=5.0, max_value=3500.0, value=450.0, step=25.0, help="Estimated transit distance.")
                distinct_sellers = st.selectbox("Distinct Sellers Count", [1, 2, 3, 4], index=0, help="Number of distinct vendors.")
                purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=datetime.datetime.now().month - 1)
                purchase_dayofweek = st.selectbox("Day of Week (0=Monday, 6=Sunday)", list(range(7)), index=0)

            st.markdown("###")
            submit = st.form_submit_button("⚡ Predict Delivery Lead Time", type="primary", use_container_width=True)

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

            st.markdown("### Result")
            st.success(f"📦 **Estimated Delivery Lead Time:** `{pred_days:.1f} days`")
            if pred_days > 20:
                st.warning("⚠️ High lead-time risk detected. Regional transit buffer recommended.")
            else:
                st.info("✅ Lead time is within standard operating benchmarks.")

# ==========================================
# PAGE 4: INVENTORY & ROUTE OPTIMIZATION
# ==========================================
elif page == "📦 Inventory & Route Optimization":
    st.title("Inventory Control & Routing Optimization")
    
    st.subheader("1. Inventory Reorder Points (ROP) & Safety Stock")
    st.markdown("Category-level safety stock calculated for a 95% service level ($Z=1.65$) with 5-day supplier lead time.")
    
    if not inventory.empty:
        st.dataframe(inventory.reset_index(drop=True), use_container_width=True, height=250)
    else:
        st.warning("⚠️ Inventory summary not found. Please run `python scripts/run_optimization.py` to generate it.")

    st.markdown("---")
    st.subheader("2. Regional Fulfillment Clusters & Route Optimization")
    st.markdown("Customer delivery locations grouped into sub-hubs, reducing route transit distances by **~45.8%** via nearest-neighbor TSP heuristics.")
    
    if os.path.exists('reports/figures/sp_delivery_clusters.png'):
        st.image('reports/figures/sp_delivery_clusters.png', caption="São Paulo Delivery Zone Optimization (K-Means Clustering)", use_container_width=True)
    else:
        st.warning("⚠️ Cluster figure not found. Please run `python scripts/run_optimization.py` to generate the image.")