# Logistics Exploratory Data Analysis (EDA) Summary

- **Dataset:** Olist Brazilian E-Commerce (orders delivered between 2016–2018)
- **Analyzed Delivered Orders:** 96,478
- **Overall On-Time Delivery Rate (OTD):** 91.88%
- **Average Delivery Time:** 12.56 days
- **Average Delay Metric:** -11.18 days *(delivered ~11 days before estimated deadline)*
- **Average Freight Cost per Order:** R$ 22.79

---

1. Core Operational Insights
* **Delivery Distribution:** Most deliveries complete within 7–18 days, with a right-skewed tail representing remote destinations and transit bottlenecks.

**Geographical Variations:** Significant lead-time and on-time performance disparities exist across Brazilian states, highlighting regional fulfillment hub dependencies.

**Freight & Distance Dynamics:** Shipping distance directly influences total freight costs, validating the value of route clustering and regional fulfillment optimization.

2. Generated Visual Artifacts
* `reports/figures/delivery_distributions.png`
* `reports/figures/daily_sales_trend.png`
* `reports/figures/otd_by_state.png`
* `reports/figures/top_categories.png`
* `reports/figures/customer_locations_map.html`