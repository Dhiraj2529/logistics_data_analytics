# Logistics Data Analysis & Optimization

A Python-based framework for analyzing and optimizing logistics operations using the **Olist Brazilian E-Commerce public dataset**. The project demonstrates how data science techniques—demand forecasting, delivery time prediction, clustering, vehicle routing, and inventory optimization—can improve supply chain efficiency and resource allocation.

## 📋 Project Overview

This repository implements an end-to-end logistics analytics pipeline:

- **Data Processing** – Cleans, merges, and enriches raw Olist tables into analysis-ready datasets.
- **Exploratory Data Analysis (EDA)** – Computes baseline KPIs (on-time delivery, average delay, cost per order) and visualizes trends, bottlenecks, and geographical patterns.
- **Predictive Modeling** – 
  - Demand forecasting using **Facebook Prophet** (30-day forecast).
  - Delivery time prediction using **XGBoost**.
- **Optimization** – 
  - Customer delivery zone clustering with **K-Means**.
  - Vehicle routing problem (CVRP) solved with **OR-Tools**.
  - Inventory safety stock and reorder point calculation.
- **Interactive Dashboard** – Built with **Streamlit** to monitor KPIs, forecasts, and optimization results.

## 🗂️ Repository Structure
.
├── data/
│ ├── raw/ # Original Olist CSV files (place them here)
│ ├── processed/ # Cleaned and derived datasets (generated)
│ └── README.md # Data dictionary and source description
├── notebooks/
│ ├── 01_data_processing.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_demand_forecasting.ipynb
│ ├── 04_delivery_time_prediction.ipynb
│ └── 05_optimization.ipynb
├── scripts/
│ ├── process_data.py # End-to-end data cleaning and feature engineering
│ └── (optional) train_models.py
├── models/ # Saved model files (.pkl)
├── dashboard/
│ └── dashboard.py # Streamlit dashboard
├── reports/
│ └── figures/ # Saved plots and maps
├── docs/
│ └── project_plan.md # Detailed planning & strategy document
├── requirements.txt
└── README.md



## 📊 Data

The analysis uses the **Olist Brazilian E-Commerce Dataset** available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Download the CSV files and place them in `data/raw/olist/`. The main tables used are:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_geolocation_dataset.csv`

Data cleaning and processing steps (including geolocation aggregation, distance calculation, and KPI computation) are automated in `scripts/process_data.py`. The processed datasets are saved in `data/processed/`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/logistics-data-analysis.git
cd logistics-data-analysis
# ---------------------------------------------------------------------------
2. INSTALL DEPENDENCIES

pip install -r requirements.txt
# ----------------------------------------------------------------------------
3. Download the Olist dataset
Go to the Kaggle dataset page and download the zip.

Extract the CSV files into data/raw/olist/.
# ----------------------------------------------------------------------------
4. Run data processing

python scripts/process_data.py
# ----------------------------------------------------------------------------
6. Launch the dashboard
streamlit run dashboard/dashboard.py
# ----------------------------------------------------------------------------
🔑 Key Performance Indicators (KPIs)
The project tracks the following logistics KPIs:

KPI	Description	Target
On-Time Delivery (OTD)	% of orders delivered on or before estimated date	≥ 95%
Average Delivery Time	Mean days from purchase to delivery	Reduce
Average Delay	Mean days late (positive = late)	Minimize
Cost per Order	Average freight value per order	Reduce by 10-15%
Vehicle Utilization	(Simulated) % of vehicle capacity used	≥ 85%
Stockout Rate	(Derived) demand variability informs safety stock	< 2%

📈 Results
Demand Forecast – Prophet model with MAPE < 20% on test set.

Delivery Time Prediction – XGBoost achieves RMSE ~X days (see notebook).

Delivery Zones – K-Means clusters customers into 8 geographical zones for balanced routing.

Route Optimization – OR-Tools CVRP reduces simulated total distance by ~15% compared to naive routing.

Inventory Optimization – Safety stock and reorder points calculated for top product categories.

📄 Documentation
The full planning and strategy document is available in docs/project_plan.md. It outlines the project definition, data science approach, strategic roadmap, and expected outcomes.

🛠️ Technologies Used
Python 3.8+

Pandas, NumPy – data manipulation

Matplotlib, Seaborn, Plotly – visualization

Scikit-learn – clustering, machine learning

Facebook Prophet – time series forecasting

XGBoost – regression

OR-Tools – vehicle routing optimization

Streamlit – dashboard deployment

Folium – geospatial mapping

📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

👤 Author
Dhiraj Gandhi
B.Tech Computer Science & Engineering
https://github.com/Dhiraj2529 | https://www.linkedin.com/in/dhiraj-gandhi-3a2545282/