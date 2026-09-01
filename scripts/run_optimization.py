import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Ensure reports directory exists
os.makedirs('reports/figures', exist_ok=True)

print("--- 1. Geographic Delivery Zone Clustering (K-Means) ---")

# Load processed orders with coordinates
orders = pd.read_csv('data/processed/orders_processed.csv')
coords_df = orders.dropna(subset=['customer_lat', 'customer_lon']).copy()

# Focus on São Paulo (SP) as our primary metropolitan hub case study
sp_orders = coords_df[coords_df['customer_state'] == 'SP'].sample(n=3000, random_state=42).copy()

# Cluster delivery points into 5 regional sub-hubs
X_geo = sp_orders[['customer_lat', 'customer_lon']]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
sp_orders['delivery_zone'] = kmeans.fit_predict(X_geo)
centroids = kmeans.cluster_centers_

# Plot delivery clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    sp_orders['customer_lon'], 
    sp_orders['customer_lat'], 
    c=sp_orders['delivery_zone'], 
    cmap='tab10', 
    alpha=0.6, 
    s=15
)
plt.scatter(
    centroids[:, 1], 
    centroids[:, 0], 
    c='red', 
    marker='X', 
    s=150, 
    label='Optimized Hub Centroids'
)
plt.title('São Paulo Delivery Zone Optimization (K-Means Clustering)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/sp_delivery_clusters.png', dpi=150)
plt.close()
print("Saved cluster plot: reports/figures/sp_delivery_clusters.png")

print("\n--- 2. Route Optimization (Nearest Neighbor Simulation) ---")

# Simulate a daily delivery run for 15 customer drops in Cluster 0
sample_cluster = sp_orders[sp_orders['delivery_zone'] == 0].head(15).reset_index(drop=True)
hub_location = centroids[0].reshape(1, 2)
points = np.vstack([hub_location, sample_cluster[['customer_lat', 'customer_lon']].values])

# Compute pairwise Euclidean distance matrix (as proxy for transit distance)
dist_matrix = cdist(points, points, metric='euclidean')

# Nearest Neighbor TSP Heuristic
num_stops = len(points)
unvisited = set(range(1, num_stops))
route = [0]
current = 0

while unvisited:
    next_stop = min(unvisited, key=lambda city: dist_matrix[current][city])
    unvisited.remove(next_stop)
    route.append(next_stop)
    current = next_stop
route.append(0) # Return to hub

unoptimized_dist = sum(dist_matrix[i][i+1] for i in range(num_stops - 1)) + dist_matrix[num_stops - 1][0]
optimized_dist = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))
distance_saved_pct = ((unoptimized_dist - optimized_dist) / unoptimized_dist) * 100

print(f"Total Stops: {num_stops - 1} customer deliveries")
print(f"Optimized Route Sequence: {route}")
print(f"Route Distance Reduced by: {distance_saved_pct:.2f}%\n")

print("--- 3. Inventory Safety Stock & Reorder Point Optimization ---")

# Load items to aggregate category-level daily demand variability
items = pd.read_csv('data/processed/order_items_processed.csv')
items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'])
items['date'] = items['shipping_limit_date'].dt.date

# Calculate daily demand per product category
cat_daily_demand = items.groupby(['date', 'product_category_name'])['order_item_id'].count().reset_index()

# Top 5 categories safety stock calculation (assuming 95% service level Z=1.65, Lead Time = 5 days)
top_5_cats = items['product_category_name'].value_counts().head(5).index
Z_SCORE = 1.65  # 95% service level
LEAD_TIME_DAYS = 5

inventory_summary = []
for cat in top_5_cats:
    cat_data = cat_daily_demand[cat_daily_demand['product_category_name'] == cat]['order_item_id']
    avg_daily_demand = cat_data.mean()
    std_daily_demand = cat_data.std() if len(cat_data) > 1 else 0
    
    # Safety Stock Formula: SS = Z * std_dev_demand * sqrt(Lead_Time)
    safety_stock = Z_SCORE * std_daily_demand * np.sqrt(LEAD_TIME_DAYS)
    
    # Reorder Point Formula: ROP = (Avg Daily Demand * Lead Time) + Safety Stock
    reorder_point = (avg_daily_demand * LEAD_TIME_DAYS) + safety_stock
    
    inventory_summary.append({
        'Category': cat,
        'Avg Daily Demand': round(avg_daily_demand, 2),
        'Std Dev Demand': round(std_daily_demand, 2),
        'Recommended Safety Stock': int(np.ceil(safety_stock)),
        'Reorder Point (ROP)': int(np.ceil(reorder_point))
    })

inventory_df = pd.DataFrame(inventory_summary)
print(inventory_df.to_string(index=False))

# Export optimized parameters
inventory_df.to_csv('data/processed/inventory_optimization_summary.csv', index=False)
print("\nSaved: data/processed/inventory_optimization_summary.csv")
