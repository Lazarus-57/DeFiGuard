import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def _resolve_user_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {
        "transaction_id",
        "tx_hash",
        "aml_label",
        "label_note",
        "split",
        "block_time",
        "from",
        "to",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Keep only numeric features for traditional models.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    num = 2 * p * r
    f1 = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom > 0)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def _recall_at_precision_target(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.80) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid_recalls = recall[precision >= target_precision]
    if len(valid_recalls) == 0:
        return 0.0
    return float(np.max(valid_recalls))


def _evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    threshold = _best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "val_pr_auc": float(average_precision_score(y_true, y_prob)),
        "val_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "val_recall_at_p80": float(_recall_at_precision_target(y_true, y_prob, target_precision=0.80)),
        "threshold": float(threshold),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Step 1: Compare LR, RF, and XGBoost on validation split.")
    parser.add_argument("--data", default="data/processed/5k/modeling_dataset_transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = _resolve_user_path(args.data)
    df = pd.read_csv(data_path)

    if "split" not in df.columns:
        raise ValueError("Input dataset must contain a split column with train/val/test.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty in the provided dataset.")

    X_train = _prepare_feature_matrix(train_df)
    X_val = _prepare_feature_matrix(val_df)
    y_train = train_df["aml_label"].astype(int).to_numpy()
    y_val = val_df["aml_label"].astype(int).to_numpy()

    if np.sum(y_train == 1) == 0:
        raise ValueError("Training split has zero positive labels; cannot train AML classifier.")

    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / pos_count

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=args.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            booster="gbtree",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            gamma=0.0,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            n_jobs=-1,
            early_stopping_rounds=50,
        ),
    }

    results = []

    for model_name, model in models.items():
        start = time.perf_counter()

        if model_name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            best_iteration = int(getattr(model, "best_iteration", -1))
        else:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            best_iteration = -1

        elapsed = time.perf_counter() - start
        metrics = _evaluate_probs(y_val, y_prob)
        metrics["model"] = model_name
        metrics["train_seconds"] = elapsed
        metrics["best_iteration"] = best_iteration
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    print("=" * 88)
    print("PHASE 1 STEP 1 - TRADITIONAL MODEL COMPARISON (VALIDATION SPLIT)")
    print("=" * 88)
    print(f"Data file: {data_path}")
    print(f"Train rows: {len(train_df)} | Positives: {int(np.sum(y_train == 1))} | Negatives: {int(np.sum(y_train == 0))}")
    print(f"Val rows: {len(val_df)} | Positives: {int(np.sum(y_val == 1))} | Negatives: {int(np.sum(y_val == 0))}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Seed: {args.seed}")
    print("-" * 88)

    display_cols = [
        "model",
        "val_pr_auc",
        "val_roc_auc",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_recall_at_p80",
        "threshold",
        "train_seconds",
        "best_iteration",
    ]

    printable = results_df[display_cols].copy()
    float_cols = [c for c in printable.columns if c not in {"model", "best_iteration"}]
    for col in float_cols:
        printable[col] = printable[col].astype(float).round(4)

    print(printable.to_string(index=False))
    print("-" * 88)

    winner = results_df.iloc[0]
    print(
        "Recommended winner for next step: "
        f"{winner['model']} (best val_pr_auc={winner['val_pr_auc']:.4f})"
    )


if __name__ == "__main__":
    main()
