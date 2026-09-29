"""Evaluate the completed Phase 1 model on its untouched 100k test split."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np

from inference import DefiGuardInference

def precision_at_k(y_true, scores, k_frac=0.01):
    k = max(1, int(len(scores) * k_frac))
    if k == 0:
        return 0.0
    top_indices = np.argsort(scores)[::-1][:k]
    return float(y_true[top_indices].mean())


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "100k" / "all_patterns" / "modeling_dataset.csv"


def main() -> None:
    print("Initializing Inference Engine...")
    infer = DefiGuardInference()

    print("Loading the untouched 100k test split...")
    df = pd.read_csv(DATA_PATH)
    test_df = df.loc[df["split"] == "test"].copy()

    print(f"Running inference on {len(test_df):,} transactions...")
    results_df = infer.predict(test_df)

    y_true = test_df["aml_label"].astype(int).to_numpy()
    y_prob = results_df["suspicion_score"].to_numpy()
    y_pred = (y_prob >= infer.threshold).astype(int)

    print("\nBlind Test Metrics")
    print(f"Test Transactions: {len(test_df):,}")
    print(f"Flagged as Suspicious: {y_pred.sum():,} ({y_pred.mean() * 100:.2f}%)")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_prob):.4f}")
    print(f"Brier Score Loss (Calibration): {brier_score_loss(y_true, y_prob):.4f}")
    print(f"Precision (Fixed Threshold): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall (Fixed Threshold): {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    print("\nAlert Budgeting Metrics (Real-World Compliance)")
    print(f"Precision @ Top 0.1% Alerts: {precision_at_k(y_true, y_prob, 0.001):.2%}")
    print(f"Precision @ Top 0.5% Alerts: {precision_at_k(y_true, y_prob, 0.005):.2%}")
    print(f"Precision @ Top 1.0% Alerts: {precision_at_k(y_true, y_prob, 0.01):.2%}")

    print("\nPattern Classification Breakdown (flagged transactions)")
    print(results_df.loc[y_pred == 1, "pattern_type"].value_counts())


if __name__ == "__main__":
    main()
