import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Load orders with coordinates
orders_path = os.path.join(PROCESSED_DIR, 'orders_processed.csv')
orders = pd.read_csv(orders_path, parse_dates=['order_purchase_timestamp',
                                               'order_delivered_customer_date',
                                               'order_estimated_delivery_date'])

# Drop missing coordinates and take a sample (optional, but speeds up clustering)
coords = orders.dropna(subset=['customer_lat', 'customer_lon'])[['customer_lat', 'customer_lon']].sample(10000, random_state=42)

# Choose number of clusters (8, as earlier)
k = 8
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(coords)

# Assign zone to orders (only for those with coordinates)
orders['delivery_zone'] = np.nan
orders.loc[coords.index, 'delivery_zone'] = kmeans.labels_

# Keep only order_id and delivery_zone
zones = orders[['order_id', 'delivery_zone']].dropna()
zones.to_csv(os.path.join(PROCESSED_DIR, 'delivery_zones.csv'), index=False)

print(f"Saved {len(zones)} zone assignments to delivery_zones.csv")