# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

# np.random.seed(42)

# # Parameters
# n_orders = 2000
# n_shipments = 1500
# n_telemetry = 3000
# start_date = datetime(2024, 1, 1)

# # ---------- Orders ----------
# orders = pd.DataFrame({
#     'order_id': range(1, n_orders + 1),
#     'order_date': [start_date + timedelta(hours=i) for i in range(n_orders)],
#     'customer_id': np.random.randint(1000, 5000, n_orders),
#     'sku': np.random.choice(['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'], n_orders, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
#     'quantity': np.random.randint(1, 15, n_orders),
#     'customer_lat': np.random.uniform(12.80, 13.20, n_orders),   # Bengaluru region
#     'customer_lon': np.random.uniform(77.40, 77.80, n_orders),
#     'promised_delivery_date': [start_date + timedelta(hours=i, days=2) for i in range(n_orders)],
# })
# orders.to_csv('data/raw/orders_sample.csv', index=False)

# # ---------- Shipments ----------
# shipments = pd.DataFrame({
#     'shipment_id': range(1, n_shipments + 1),
#     'order_id': np.random.choice(orders['order_id'], n_shipments, replace=False),
#     'vehicle_id': np.random.choice(['VAN01', 'VAN02', 'TRUCK01', 'TRUCK02', 'TRUCK03'], n_shipments),
#     'planned_departure': [start_date + timedelta(hours=i, days=2) for i in range(n_shipments)],
#     'actual_departure': [start_date + timedelta(hours=i + np.random.randint(-2, 4), days=2) for i in range(n_shipments)],
#     'distance_km': np.random.uniform(5, 80, n_shipments).round(2),
#     'fuel_cost': np.random.uniform(200, 1500, n_shipments).round(2),
#     'delivery_status': np.random.choice(['Delivered', 'Delayed', 'In Transit'], n_shipments, p=[0.85, 0.1, 0.05]),
# })
# shipments.to_csv('data/raw/shipments_sample.csv', index=False)

# # ---------- Telemetry ----------
# telemetry = pd.DataFrame({
#     'telemetry_id': range(1, n_telemetry + 1),
#     'vehicle_id': np.random.choice(['VAN01', 'VAN02', 'TRUCK01', 'TRUCK02', 'TRUCK03'], n_telemetry),
#     'timestamp': [start_date + timedelta(minutes=i*5) for i in range(n_telemetry)],
#     'speed_kmh': np.random.normal(45, 15, n_telemetry).clip(0, 100),
#     'idle_minutes': np.random.exponential(5, n_telemetry).round(1),
#     'stop_count': np.random.poisson(3, n_telemetry),
#     'latitude': np.random.uniform(12.80, 13.20, n_telemetry),
#     'longitude': np.random.uniform(77.40, 77.80, n_telemetry),
# })
# telemetry.to_csv('data/raw/telemetry_sample.csv', index=False)

# # ---------- Inventory ----------
# inventory = pd.DataFrame({
#     'sku': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005'],
#     'current_stock': np.random.randint(50, 500, 5),
#     'reorder_point': np.random.randint(100, 300, 5),
#     'lead_time_days': [3, 5, 7, 10, 14],
#     'unit_cost': np.random.uniform(10, 200, 5).round(2),
#     'holding_cost_per_unit_day': np.random.uniform(0.1, 2.0, 5).round(3),
# })
# inventory.to_csv('data/raw/inventory_sample.csv', index=False)

# print("Sample data files generated in data/raw/")