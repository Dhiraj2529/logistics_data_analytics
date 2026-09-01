import pandas as pd
import numpy as np

# Core tables
orders = pd.read_csv('data/raw/olist/olist_orders_dataset.csv')
order_items = pd.read_csv('data/raw/olist/olist_order_items_dataset.csv')
customers = pd.read_csv('data/raw/olist/olist_customers_dataset.csv')
sellers = pd.read_csv('data/raw/olist/olist_sellers_dataset.csv')
products = pd.read_csv('data/raw/olist/olist_products_dataset.csv')
geolocation = pd.read_csv('data/raw/olist/olist_geolocation_dataset.csv')

# clean and process timestamps
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# For delivery time analysis, keep only orders with delivered date
orders_delivered = orders[orders['order_status'] == 'delivered'].copy()

# Merge with customers to get customer zip code
orders_cust = orders_delivered.merge(
    customers[['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']],
    on='customer_id',
    how='left'
)

orders_cust['delivery_time_days'] = (
    orders_cust['order_delivered_customer_date'] - orders_cust['order_purchase_timestamp']
).dt.total_seconds() / 86400

orders_cust['delay_days'] = (
    orders_cust['order_delivered_customer_date'] - orders_cust['order_estimated_delivery_date']
).dt.total_seconds() / 86400

# creating two processed datasets: one for delivery time analysis and another for delay analysis
# ---1---
orders_processed = orders_cust[[
    'order_id', 'customer_id', 'order_purchase_timestamp',
    'order_delivered_customer_date', 'order_estimated_delivery_date',
    'delivery_time_days', 'delay_days',
    'customer_zip_code_prefix', 'customer_city', 'customer_state'
]]

orders_processed.to_csv('data/processed/orders_processed.csv', index=False)
# ---2---
# Merge order_items with orders for timestamps and customer info
items = order_items.merge(
    orders_delivered[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_delivered_customer_date']],
    on='order_id',
    how='inner'
)

# Add customer zip
items = items.merge(
    customers[['customer_id', 'customer_zip_code_prefix']],
    on='customer_id',
    how='left'
)

# Add seller zip
items = items.merge(
    sellers[['seller_id', 'seller_zip_code_prefix']],
    on='seller_id',
    how='left'
)

# Add product category (optional)
items = items.merge(
    products[['product_id', 'product_category_name']],
    on='product_id',
    how='left'
)

# Save
items.to_csv('data/processed/order_items_processed.csv', index=False)

# --------------------------------------------
# Geolocation: Aggregate Zip Code Coordinates
# --------------------------------------------
geo_agg = geolocation.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()
geo_agg.columns = ['zip_code_prefix', 'lat', 'lon']

# Save for later use
geo_agg.to_csv('data/processed/zip_coordinates.csv', index=False)

# For orders: customer coordinates
orders_processed = orders_processed.merge(
    geo_agg.rename(columns={'zip_code_prefix': 'customer_zip_code_prefix', 'lat': 'customer_lat', 'lon': 'customer_lon'}),
    on='customer_zip_code_prefix',
    how='left'
)

# For items: customer and seller coordinates
items = items.merge(
    geo_agg.rename(columns={'zip_code_prefix': 'customer_zip_code_prefix', 'lat': 'customer_lat', 'lon': 'customer_lon'}),
    on='customer_zip_code_prefix',
    how='left'
)
items = items.merge(
    geo_agg.rename(columns={'zip_code_prefix': 'seller_zip_code_prefix', 'lat': 'seller_lat', 'lon': 'seller_lon'}),
    on='seller_zip_code_prefix',
    how='left'
)

# Save updated versions
orders_processed.to_csv('data/processed/orders_processed.csv', index=False)
items.to_csv('data/processed/order_items_processed.csv', index=False)

from math import radians, sin, cos, sqrt, asin

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# Apply vectorized for speed (use numpy)
items['seller_to_customer_km'] = np.vectorize(haversine_km)(
    items['seller_lat'], items['seller_lon'],
    items['customer_lat'], items['customer_lon']
)

items.to_csv('data/processed/order_items_processed.csv', index=False)

# ----------------------------------
# Demand dataset for forcasting
# ----------------------------------
# Daily number of items sold per category
daily_demand = items.groupby([
    pd.Grouper(key='order_purchase_timestamp', freq='D'),
    'product_category_name'
]).size().reset_index(name='quantity')

daily_demand.to_csv('data/processed/daily_demand_by_category.csv', index=False)

daily_total = items.groupby(pd.Grouper(key='order_purchase_timestamp', freq='D')).agg({
    'price': 'sum',
    'order_id': 'count'
}).reset_index()
daily_total.columns = ['date', 'revenue', 'orders']
daily_total.to_csv('data/processed/daily_sales.csv', index=False)


