import os
from pathlib import Path
import pandas as pd
from inference import DefiGuardInference

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "25k"
REPORTS_DIR = SCRIPT_DIR.parent / "Phase 1 Reports"

def main():
    print("Initializing Inference Engine...")
    try:
        infer = DefiGuardInference()
    except Exception as e:
        print(f"Failed to load models. Did you run phase1_master_training.py? Error: {e}")
        return

    print("Loading 25k Out-Of-Sample Dataset...")
    data_path = DATA_DIR / "augmented_transactions_multipattern.csv"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return
        
    df_25k = pd.read_csv(data_path)
    
    print(f"Running inference on {len(df_25k)} transactions...")
    results_df = infer.predict(df_25k)
    
    print("Inference Complete! Analyzing results...")
    
    # Calculate some quick statistics
    total_tx = len(results_df)
    flagged = results_df['aml_flag'].sum()
    flagged_pct = (flagged / total_tx) * 100
    
    print(f"\nTotal Transactions Processed: {total_tx}")
    print(f"Flagged as Suspicious: {flagged} ({flagged_pct:.2f}%)")
    
    print("\nPattern Classification Breakdown:")
    print(results_df[results_df['aml_flag'] == 1]['pattern_type'].value_counts())
    
    print("\nTop SHAP Reasons for Suspicion:")
    print(results_df[results_df['aml_flag'] == 1]['top_shap_reason'].value_counts().head(5))
    
    # Save results
    output_path = REPORTS_DIR / "25k_inference_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full results to {output_path}")

if __name__ == "__main__":
    main()
