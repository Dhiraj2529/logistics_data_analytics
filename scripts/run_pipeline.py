import os
import subprocess
import sys

def run_full_pipeline():
    print("🚀 Starting automated supply chain data pipeline...")
    
    # Step 1: Check if new raw data exists
    raw_file = 'data/raw/new_incoming_orders.csv'
    if not os.path.exists(raw_file):
        print(f"⚠️ Warning: {raw_file} not found. Running with existing data structures.")
    
    # Step 2: Execute data processing
    print("\n[1/3] Processing raw data and calculating delivery/delay metrics...")
    try:
        from scripts.process_custom_data import process_uploaded_orders
        if os.path.exists(raw_file):
            process_uploaded_orders(raw_file)
        else:
            print("➡️ Skipping custom file processing (using defaults).")
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return

    # Step 3: Re-train ML models (Prophet Demand Forecasting & XGBoost Lead Time)
    print("\n[2/3] Re-training machine learning models (XGBoost & Prophet)...")
    try:
        if os.path.exists('scripts/train_models.py'):
            subprocess.run([sys.executable, 'scripts/train_models.py'], check=True)
            print("✅ Models re-trained successfully.")
        else:
            print("⚠️ scripts/train_models.py not found.")
    except Exception as e:
        print(f"❌ Error training models: {e}")

    # Step 4: Re-run inventory optimization (Safety Stock & ROP)
    print("\n[3/3] Recalculating Safety Stock and Reorder Points (ROP)...")
    try:
        if os.path.exists('scripts/run_optimization.py'):
            subprocess.run([sys.executable, 'scripts/run_optimization.py'], check=True)
            print("✅ Inventory optimization summary updated.")
        else:
            print("⚠️ scripts/run_optimization.py not found.")
    except Exception as e:
        print(f"❌ Error in inventory optimization: {e}")

    print("\n🎉 Pipeline execution complete! Your dashboard is fully synced with the new data.")

if __name__ == "__main__":
    run_full_pipeline()