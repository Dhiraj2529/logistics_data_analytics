import pandas as pd
import numpy as np
import os

def process_uploaded_orders(file_path_or_buffer):
    """
    Cleans and processes raw order data to compute required metrics 
    like delivery_time_days and delay_days.
    """
    print("🔄 Processing custom dataset...")
    df = pd.read_csv(file_path_or_buffer)
    
    # 1. Ensure date columns are parsed correctly
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # 2. Drop rows missing critical timestamps if any
    if 'order_purchase_timestamp' in df.columns and 'order_delivered_customer_date' in df.columns:
        df = df.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])
        
    # 3. Compute Core Logistics Metrics (Feature Engineering)
    if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400.0
        
    if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
        df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86400.0
        
    # 4. Fallbacks for missing categorical/numerical columns required by models
    if 'customer_state' not in df.columns:
        df['customer_state'] = 'SP' # Default fallback state
        
    if 'freight_value' not in df.columns:
        df['freight_value'] = 20.0 # Default fallback freight
        
    if 'price' not in df.columns:
        df['price'] = 100.0 # Default fallback price
        
    # Save processed version locally for the dashboard to use
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/orders_processed.csv', index=False)
    
    # Generate daily sales summary for Prophet / Time-series chart
    if 'order_purchase_timestamp' in df.columns:
        df['date'] = df['order_purchase_timestamp'].dt.date
        daily_sales = df.groupby('date').size().reset_index(name='orders')
        daily_sales.to_csv('data/processed/daily_sales.csv', index=False)
        
    print("✅ Custom data processing complete! Processed files saved.")
    return df

if __name__ == "__main__":
    # Test execution if run directly
    input_file = 'data/raw/new_incoming_orders.csv' # or your custom path
    if os.path.exists(input_file):
        process_uploaded_orders(input_file)
    else:
        print("⚠️ Place your raw CSV in data/raw/ or upload via Streamlit.")