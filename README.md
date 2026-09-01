# Logistics Data Analysis & Optimization

This repository contains a Python implementation of a data-driven logistics strategy for a regional distribution network. It includes demand forecasting, delivery zone clustering, vehicle routing optimization, inventory safety stock calculation, and anomaly detection.

## Features
- SKU-level demand forecasting using Prophet and XGBoost.
- K-Means clustering for delivery zone creation.
- VRP solver with OR-Tools.
- Dynamic safety stock and reorder point calculator.
- Real-time anomaly detection with Isolation Forest.

## Quick Start
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebooks in the `notebooks/` folder or execute the scripts in `scripts/`.
4. To use real data, download from [(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)e] and place in data/raw/ before running the notebooks.