# Logistics Exploratory Data Analysis (EDA) Summary

## 1. Baseline Performance Indicators
* **Analyzed Delivered Orders:** 96,478
* **On-Time Delivery Rate (OTD):** 91.88%
* **Average Delivery Lead Time:** 12.56 days
* **Average Delay Metric:** -11.18 days *(delivered ~11 days before estimated delivery deadline)*
* **Average Freight Cost per Order:** R$ 22.79

## 2. Core Operational Insights
* **Delivery Distribution:** The vast majority of deliveries complete within 7–18 days, with a right-skewed long tail representing remote regions and outlier logistical bottlenecks.
* **Geographical Variations:** Significant lead-time and on-time performance disparities exist across Brazilian states, highlighting regional fulfillment hub dependencies.
* **Freight & Distance Dynamics:** Shipping distance directly influences total freight costs, reinforcing the value of route clustering and regional fulfillment optimization.

## 3. Generated Visual Artifacts
* `reports/figures/delivery_distributions.png`
* `reports/figures/daily_sales_trend.png`
* `reports/figures/otd_by_state.png`
* `reports/figures/top_categories.png`
* `reports/figures/customer_locations_map.html`