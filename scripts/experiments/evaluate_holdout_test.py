"""
Evaluate DeFIGuard's Blind Predictions on the 100k Holdout Test Set against the Secret Ground Truth Key.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_PATH = (PROJECT_ROOT / "Phase 1" / "scripts").resolve()
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from inference import DefiGuardInference

BLINDED_CSV = PROJECT_ROOT / "data" / "samples" / "holdout_test_blinded.csv"
KEY_CSV = PROJECT_ROOT / "data" / "samples" / "holdout_ground_truth_key.csv"

def main():
    print("================================================================================")
    print(" DEFIGUARD BLIND BENCHMARK EVALUATION (100K HOLDOUT TEST SPLIT)")
    print("================================================================================")
    
    # 1. Load Ground Truth Key
    key_df = pd.read_csv(KEY_CSV)
    print(f"Loaded Secret Ground Truth Key: {len(key_df):,} transactions")
    print(f"True Class Balance -> Laundering: {(key_df['aml_label'] == 1).sum():,} | Normal: {(key_df['aml_label'] == 0).sum():,}")
    
    # 2. Run Inference on Blinded File
    print(f"\nRunning DeFIGuard ML Engine on {BLINDED_CSV.name} (Zero Labels Provided)...")
    engine = DefiGuardInference()
    blind_df = pd.read_csv(BLINDED_CSV)
    
    results_df, norm_meta = engine.predict(blind_df, compute_detailed_shap=False, return_metadata=True)
    print(f"Inference Completed! Total scored transactions: {len(results_df):,}")
    print(f"Optimal Decision Threshold in Engine: {engine.threshold:.3f}")
    
    # 3. Join predictions with Ground Truth Key on tx_hash
    merged = results_df.merge(key_df, on='tx_hash', how='inner')
    
    y_true = merged['aml_label'].values
    y_prob = merged['suspicion_score'].values
    y_pred = (y_prob >= engine.threshold).astype(int)
    
    # 4. Compute Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    print("\n--------------------------------------------------------------------------------")
    print(" CONFUSION MATRIX & VERDICT")
    print("--------------------------------------------------------------------------------")
    print(f"  True Positives  (Laundering Caught):     {tp:>5} / {tp+fn}")
    print(f"  False Negatives (Laundering Missed):     {fn:>5} / {tp+fn}")
    print(f"  True Negatives  (Normal Cleared):        {tn:>5} / {tn+fp} ({tn/(tn+fp)*100:.1f}%)")
    print(f"  False Positives (Normal Flagged):        {fp:>5} / {tn+fp}")
    print("--------------------------------------------------------------------------------")
    print(f"  RECALL (Detection Rate):                 {rec*100:>6.2f}%")
    print(f"  PRECISION (Alert Accuracy):              {prec*100:>6.2f}%")
    print(f"  F1-SCORE:                                {f1:>6.4f}")
    print(f"  ROC-AUC SCORE:                           {roc_auc:>6.4f}")
    print(f"  PR-AUC (Precision-Recall AUC):           {pr_auc:>6.4f}")
    
    # Precision @ Top K
    print("\n--------------------------------------------------------------------------------")
    print(" PRECISION @ K (RISK-BASED INVESTIGATION TRIAGE)")
    print("--------------------------------------------------------------------------------")
    sorted_df = merged.sort_values(by='suspicion_score', ascending=False).reset_index(drop=True)
    for k in [10, 25, 50, 100]:
        top_k = sorted_df.head(k)
        k_prec = top_k['aml_label'].mean()
        c = int(top_k['aml_label'].sum())
        print(f"  Precision @ Top {k:>3} Alerts:            {k_prec*100:>6.2f}% ({c}/{k} confirmed laundering flows)")
    print("================================================================================\n")

if __name__ == "__main__":
    main()
