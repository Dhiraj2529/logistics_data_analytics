import pandas as pd

orders = pd.read_csv('data/processed/orders_processed.csv')
items = pd.read_csv('data/processed/order_items_processed.csv')
daily_sales = pd.read_csv('data/processed/daily_sales.csv')

print("--- Dataset Verification ---")
print(f"Total Orders: {orders.shape[0]:,}")
print(f"Total Order Items: {items.shape[0]:,}")
print(f"Average Delivery Time: {orders['delivery_time_days'].mean():.2f} days")
print(f"Late Delivery Rate: {(orders['delay_days'] > 0).mean() * 100:.2f}%")
print(f"Average Distance: {items['seller_to_customer_km'].mean():.2f} km")
print(f"Total Processed Revenue: R$ {daily_sales['revenue'].sum():,.2f}")