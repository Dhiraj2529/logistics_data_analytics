import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Load Processed Datasets
orders = pd.read_csv(
    'data/processed/orders_processed.csv',
    parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
)
items = pd.read_csv('data/processed/order_items_processed.csv')
daily_sales = pd.read_csv('data/processed/daily_sales.csv', parse_dates=['date'])

# Print first few rows and column types
print("--- Orders Head ---")
print(orders.head())

print("\n--- Orders Info ---")
orders.info()

# --- 2. Baseline KPI Calculations ---

# a. On-Time Delivery Rate (OTD)
orders['on_time'] = orders['delay_days'] <= 0
otd_rate = orders['on_time'].mean() * 100

# b. Average Delivery Time and Delay
avg_delivery_time = orders['delivery_time_days'].mean()
avg_delay = orders['delay_days'].mean()

# c. Average Freight Cost per Order
order_freight = items.groupby('order_id')['freight_value'].sum().reset_index()
order_cost = orders[['order_id']].merge(order_freight, on='order_id', how='left')
avg_cost_per_order = order_cost['freight_value'].mean()

print("\n--- BASELINE LOGISTICS KPIS ---")
print(f"On-Time Delivery Rate: {otd_rate:.2f}%")
print(f"Average Delivery Time: {avg_delivery_time:.2f} days")
print(f"Average Delay: {avg_delay:.2f} days")
print(f"Average Freight Cost per Order: R$ {avg_cost_per_order:.2f}")

# d. Inspect Demand Trend Data
print("\n--- 2.d Demand Trend Head ---")
print(daily_sales.head())

# --- 3. Visualizations ---
sns.set_style('whitegrid')

# 3.a Distribution of Delivery Time and Delay
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(orders['delivery_time_days'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Delivery Time (days)')

sns.histplot(orders['delay_days'], bins=50, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribution of Delay (days)')
plt.tight_layout()
plt.savefig('reports/figures/delivery_distributions.png', dpi=150)
plt.close()
print("Saved: reports/figures/delivery_distributions.png")

# 3.b Time Series of Daily Orders / Revenue
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(daily_sales['date'], daily_sales['orders'], color='blue', label='Orders')
ax1.set_ylabel('Number of Orders', color='blue')

ax2 = ax1.twinx()
ax2.plot(daily_sales['date'], daily_sales['revenue'], color='green', label='Revenue', alpha=0.7)
ax2.set_ylabel('Revenue (R$)', color='green')

plt.title('Daily Sales Over Time')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('reports/figures/daily_sales_trend.png', dpi=150)
plt.close()
print("Saved: reports/figures/daily_sales_trend.png")

# 3.c On-Time Delivery by Customer State
state_performance = orders.groupby('customer_state').agg(
    total_orders=('order_id', 'count'),
    on_time_orders=('on_time', 'sum')
).reset_index()
state_performance['otd_rate'] = (state_performance['on_time_orders'] / state_performance['total_orders']) * 100
state_performance = state_performance.sort_values('otd_rate', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=state_performance, x='customer_state', y='otd_rate', order=state_performance['customer_state'])
plt.title('On-Time Delivery Rate by Customer State')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('reports/figures/otd_by_state.png', dpi=150)
plt.close()
print("Saved: reports/figures/otd_by_state.png")

# 3.d Demand by Product Category (Top 10)
top_categories = items['product_category_name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_categories.values, 
    y=top_categories.index, 
    hue=top_categories.index, 
    legend=False, 
    palette='viridis'
)
plt.title('Top 10 Product Categories by Number of Items Sold')
plt.xlabel('Number of Items')
plt.tight_layout()
plt.savefig('reports/figures/top_categories.png', dpi=150)
plt.close()
print("Saved: reports/figures/top_categories.png")

# 3.e Geographical Scatter (Customer Locations)
sample_orders = orders.dropna(subset=['customer_lat', 'customer_lon']).sample(1000, random_state=42)

# Backward-compatible scatter map call
if hasattr(px, 'scatter_map'):
    fig = px.scatter_map(
        sample_orders,
        lat='customer_lat',
        lon='customer_lon',
        hover_name='customer_city',
        zoom=3
    )
else:
    fig = px.scatter_mapbox(
        sample_orders,
        lat='customer_lat',
        lon='customer_lon',
        hover_name='customer_city',
        zoom=3,
        mapbox_style='carto-positron'
    )

fig.update_layout(title='Sample of Customer Locations')
fig.write_html('reports/figures/customer_locations_map.html')
print("Saved: reports/figures/customer_locations_map.html")

# --- 4. Correlation & Bottleneck Analysis ---
print("\n--- 4. Distance vs Freight Cost Correlation ---")
distance_freight_corr = items[['seller_to_customer_km', 'freight_value']].corr()
print(distance_freight_corr)

# Worst 5 States by On-Time Delivery Rate
worst_states = state_performance.tail(5)
print("\n--- Worst 5 States by On-Time Delivery Rate ---")
print(worst_states[['customer_state', 'total_orders', 'otd_rate']].to_string(index=False))