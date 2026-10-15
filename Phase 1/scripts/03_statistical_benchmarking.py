"""
Phase 1 - Script 03: Formal Statistical Benchmarking & Baselines (Change 5)
=============================================================================
Performs formal evaluation, temporal validation, and statistical error estimation:
1. 5-Fold Rolling Temporal Cross-Validation (expanding window, zero future data leakage).
2. Rule-Based AML Heuristic Baselines:
   - High Velocity Rule: Rapid out-flow frequency in a tight temporal window.
   - Fan-Out / Structuring Rule: High out-degree dispersion with split amounts.
   - Combined Heuristic Rule.
3. 95% Bootstrap Confidence Intervals (N=1,000) for ROC-AUC, PR-AUC, Precision@K, Brier Score.
4. Output & Artifact Generation:
   - Exports results to Phase 1 Reports Hardened/statistical_benchmarking_results.json
   - Formatted Markdown summary comparison table.
"""
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "100k_hardened" / "modeling_dataset.csv"
MODELS_DIR = SCRIPT_DIR.parent / "Phase 1 Models Hardened"
REPORTS_DIR = SCRIPT_DIR.parent / "Phase 1 Reports Hardened"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float = 0.01) -> float:
    k = max(1, int(len(scores) * k_frac))
    top_indices = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top_indices]))


def assemble_features(df: pd.DataFrame, feature_names: list, gnn_dict: dict, nts_dict: dict) -> pd.DataFrame:
    """Assembles all 51 features matching Phase 1 Models Hardened exact specification."""
    # Base features
    base_cols = [c for c in df.columns if c in feature_names]
    X = df[base_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # GNN embeddings
    wallet_to_idx = gnn_dict["wallet_to_idx"]
    embeddings = gnn_dict["embeddings"]
    emb_dim = embeddings.shape[1]

    from_wallets = df["from"].fillna("").astype(str)
    to_wallets = df["to"].fillna("").astype(str)

    from_indices = np.array([wallet_to_idx.get(w, -1) for w in from_wallets])
    to_indices = np.array([wallet_to_idx.get(w, -1) for w in to_wallets])

    zero_emb = np.zeros(emb_dim, dtype=np.float32)
    from_embs = np.where(from_indices[:, None] >= 0, embeddings[np.maximum(0, from_indices)], zero_emb)
    to_embs = np.where(to_indices[:, None] >= 0, embeddings[np.maximum(0, to_indices)], zero_emb)

    for i in range(emb_dim):
        X[f"gnn_from_emb_{i}"] = from_embs[:, i]
    for i in range(emb_dim):
        X[f"gnn_to_emb_{i}"] = to_embs[:, i]

    # NTS temporal features
    from_nts = from_wallets.map(nts_dict).fillna(0.0).astype(float).values
    to_nts = to_wallets.map(nts_dict).fillna(0.0).astype(float).values

    X["from_nts"] = from_nts
    X["to_nts"] = to_nts
    X["nts_max"] = np.maximum(from_nts, to_nts)
    X["nts_mean"] = (from_nts + to_nts) / 2.0

    # Ensure exact column ordering
    return X[feature_names].copy()


def run_rule_based_heuristics(df: pd.DataFrame, feature_df: pd.DataFrame) -> dict:
    """
    Computes heuristic rule scores:
    1. High Velocity Rule: flags accounts with high NTS burstiness + high flow ratio.
    2. Fan-out / Structuring Rule: flags high out-degree dispersion with non-trivial out-degree and flow.
    3. Combined Rule: maximum of both normalized heuristic signals.
    """
    to_nts = feature_df["to_nts"].values
    from_nts = feature_df["from_nts"].values
    nts_max = np.maximum(to_nts, from_nts)
    to_flow = feature_df["to_flow_ratio"].values
    from_flow = feature_df["from_flow_ratio"].values
    from_out_deg = feature_df["from_out_degree"].values

    # Velocity score (0.0 to 1.0 rank/percentile)
    # Higher NTS and high flow ratio indicate bursty transaction clustering
    velocity_score = nts_max * 0.7 + to_flow * 0.3

    # Structuring / Fan-out score
    # Log-scaled out-degree bounded between 0 and 1, modulated by from_flow_ratio
    deg_log = np.log1p(np.maximum(0, from_out_deg))
    deg_norm = deg_log / (deg_log.max() + 1e-6)
    fan_out_score = deg_norm * 0.6 + from_flow * 0.4

    # Combined composite rule
    combined_score = np.maximum(velocity_score, fan_out_score)

    return {
        "Rule: High Velocity": velocity_score,
        "Rule: Fan-Out / Structuring": fan_out_score,
        "Rule: Combined AML Heuristic": combined_score,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Evaluates discrimination, calibration, and alert budgeting metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision_at_0.1pct": precision_at_k(y_true, y_prob, 0.001),
        "precision_at_0.5pct": precision_at_k(y_true, y_prob, 0.005),
        "precision_at_1.0pct": precision_at_k(y_true, y_prob, 0.01),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42
) -> dict:
    """
    Computes 95% non-parametric bootstrap confidence intervals (2.5th to 97.5th percentile).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    boot_roc_auc = []
    boot_pr_auc = []
    boot_p_01 = []
    boot_p_05 = []
    boot_p_10 = []
    boot_brier = []

    k_01 = max(1, int(n_samples * 0.001))
    k_05 = max(1, int(n_samples * 0.005))
    k_10 = max(1, int(n_samples * 0.01))

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        # Guard against zero-positive boot samples
        if y_b.sum() == 0 or y_b.sum() == n_samples:
            continue

        boot_roc_auc.append(roc_auc_score(y_b, p_b))
        boot_pr_auc.append(average_precision_score(y_b, p_b))

        # Precision@K
        order = np.argsort(p_b)[::-1]
        boot_p_01.append(float(np.mean(y_b[order[:k_01]])))
        boot_p_05.append(float(np.mean(y_b[order[:k_05]])))
        boot_p_10.append(float(np.mean(y_b[order[:k_10]])))
        boot_brier.append(brier_score_loss(y_b, np.clip(p_b, 0.0, 1.0)))

    def get_ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower_2.5%": float(np.percentile(arr, 2.5)),
            "ci_upper_97.5%": float(np.percentile(arr, 97.5)),
        }

    return {
        "roc_auc": get_ci(boot_roc_auc),
        "pr_auc": get_ci(boot_pr_auc),
        "precision_at_0.1pct": get_ci(boot_p_01),
        "precision_at_0.5pct": get_ci(boot_p_05),
        "precision_at_1.0pct": get_ci(boot_p_10),
        "brier_score": get_ci(boot_brier),
    }


def run_temporal_cv(df_sorted: pd.DataFrame, X_all: pd.DataFrame, n_splits: int = 5) -> list:
    """
    5-Fold Rolling / Expanding Window Temporal Cross-Validation.
    Fold i trains on chunks 0..i and tests strictly on chunk i+1.
    Guarantees absolute forward-chaining with zero future-data leakage.
    """
    total_len = len(df_sorted)
    chunk_size = total_len // (n_splits + 1)
    fold_results = []

    print(f"\n--- Running {n_splits}-Fold Rolling Temporal Cross-Validation ---")
    y_all = df_sorted["aml_label"].astype(int).values

    for fold in range(1, n_splits + 1):
        train_end = chunk_size * fold
        test_end = min(total_len, chunk_size * (fold + 1)) if fold < n_splits else total_len

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(train_end, test_end)

        X_tr = X_all.iloc[train_indices]
        y_tr = y_all[train_indices]
        X_te = X_all.iloc[test_indices]
        y_te = y_all[test_indices]

        # Check classes present
        if y_tr.sum() == 0 or y_te.sum() == 0:
            print(f"Fold {fold}: Skipping due to zero positives in split.")
            continue

        neg_c = float(np.sum(y_tr == 0))
        pos_c = float(np.sum(y_tr == 1))

        # Train fast temporal model
        fold_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            booster="gbtree",
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg_c / pos_c,
            random_state=42 + fold,
            n_jobs=-1,
        )
        fold_model.fit(X_tr, y_tr)

        # Calibrate using the latter 20% of train chunk (temporal prefit)
        val_split_point = int(len(X_tr) * 0.8)
        X_cal_fit = X_tr.iloc[:val_split_point]
        y_cal_fit = y_tr[:val_split_point]
        X_cal_val = X_tr.iloc[val_split_point:]
        y_cal_val = y_tr[val_split_point:]

        cal_model = CalibratedClassifierCV(fold_model, method="isotonic", cv="prefit")
        cal_model.fit(X_cal_val, y_cal_val)

        probs_te = cal_model.predict_proba(X_te)[:, 1]
        fold_metrics = compute_metrics(y_te, probs_te, threshold=0.20)
        fold_metrics["fold"] = fold
        fold_metrics["train_size"] = int(len(train_indices))
        fold_metrics["test_size"] = int(len(test_indices))
        fold_metrics["test_positives"] = int(y_te.sum())

        fold_results.append(fold_metrics)
        print(
            f"Fold {fold} (Train: {len(train_indices)}, Test: {len(test_indices)}): "
            f"ROC-AUC = {fold_metrics['roc_auc']:.4f}, "
            f"PR-AUC = {fold_metrics['pr_auc']:.4f}, "
            f"P@0.1% = {fold_metrics['precision_at_0.1pct']:.2%}, "
            f"P@0.5% = {fold_metrics['precision_at_0.5pct']:.2%}"
        )

    return fold_results


def main():
    start_time = time.time()
    print("=" * 80)
    print("DEFIGUARD: FORMAL STATISTICAL BENCHMARKING & BASELINES (CHANGE 5)")
    print("=" * 80)

    # 1. Load Hardened Model Artifacts
    print(f"Loading hardened artifacts from {MODELS_DIR}...")
    with open(MODELS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open(MODELS_DIR / "gnn_wallet_embeddings.pkl", "rb") as f:
        gnn_dict = pickle.load(f)
    with open(MODELS_DIR / "nts_map.pkl", "rb") as f:
        nts_dict = pickle.load(f)
    with open(MODELS_DIR / "master_calibrated_model.pkl", "rb") as f:
        master_calibrated = pickle.load(f)
    with open(MODELS_DIR / "decision_threshold.json", "r") as f:
        threshold_info = json.load(f)
        decision_threshold = threshold_info.get("threshold", 0.20)

    # 2. Load and Chronologically Sort Hardened Dataset
    print(f"Loading modeling dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total transactions: {len(df):,}")

    # Ensure chronological order
    df["block_time_dt"] = pd.to_datetime(df["block_time"])
    df_sorted = df.sort_values(by=["block_time_dt", "block_number"]).reset_index(drop=True)
    print(f"Time range: {df_sorted['block_time_dt'].min()} -> {df_sorted['block_time_dt'].max()}")

    # Assemble all 51 features
    print("Assembling 51-feature matrix for all transactions...")
    X_all = assemble_features(df_sorted, feature_names, gnn_dict, nts_dict)
    y_all = df_sorted["aml_label"].astype(int).values

    # Test Split Extraction (for formal baseline & bootstrap evaluation)
    test_mask = df_sorted["split"] == "test"
    df_test = df_sorted[test_mask].reset_index(drop=True)
    X_test = X_all[test_mask].reset_index(drop=True)
    y_test = y_all[test_mask]
    print(f"Test split size: {len(y_test):,} transactions ({y_test.sum():,} positive AML)")

    # 3. Model Scoring & Rule-Based Baselines
    print("\n--- Scoring Tuned Calibrated XGBoost on Test Set ---")
    xgb_probs = master_calibrated.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_probs, threshold=decision_threshold)

    print("\n--- Computing Rule-Based Heuristic Baselines on Test Set ---")
    rule_scores = run_rule_based_heuristics(df_test, X_test)
    baseline_metrics = {}
    for name, scores in rule_scores.items():
        m = compute_metrics(y_test, scores, threshold=0.5)
        baseline_metrics[name] = m
        print(
            f"{name:32s}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
            f"P@0.1%={m['precision_at_0.1pct']:.2%}, P@0.5%={m['precision_at_0.5pct']:.2%}, Recall={m['recall']:.2%}"
        )

    # 4. Bootstrap Confidence Intervals (N=1,000)
    print("\n--- Computing 95% Bootstrap Confidence Intervals (N=1,000) ---")
    print("Bootstrapping Calibrated XGBoost...")
    xgb_ci = bootstrap_confidence_intervals(y_test, xgb_probs, n_bootstraps=1000)

    baseline_ci = {}
    for name, scores in rule_scores.items():
        print(f"Bootstrapping {name}...")
        baseline_ci[name] = bootstrap_confidence_intervals(y_test, scores, n_bootstraps=1000)

    # 5. 5-Fold Rolling Temporal Cross-Validation
    temporal_folds = run_temporal_cv(df_sorted, X_all, n_splits=5)
    mean_temp_roc = float(np.mean([f["roc_auc"] for f in temporal_folds]))
    std_temp_roc = float(np.std([f["roc_auc"] for f in temporal_folds]))
    mean_temp_pr = float(np.mean([f["pr_auc"] for f in temporal_folds]))
    std_temp_pr = float(np.std([f["pr_auc"] for f in temporal_folds]))
    mean_temp_p01 = float(np.mean([f["precision_at_0.1pct"] for f in temporal_folds]))
    mean_temp_p05 = float(np.mean([f["precision_at_0.5pct"] for f in temporal_folds]))

    print(f"\nRolling Temporal 5-Fold Average: ROC-AUC = {mean_temp_roc:.4f} (±{std_temp_roc:.4f}), PR-AUC = {mean_temp_pr:.4f} (±{std_temp_pr:.4f})")

    # 6. Export Results
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset": "100k_hardened",
        "total_transactions": len(df_sorted),
        "test_transactions": len(df_test),
        "decision_threshold": decision_threshold,
        "calibrated_xgboost": {
            "point_metrics": xgb_metrics,
            "bootstrap_95ci": xgb_ci,
        },
        "rule_based_baselines": {
            name: {
                "point_metrics": baseline_metrics[name],
                "bootstrap_95ci": baseline_ci[name],
            }
            for name in rule_scores
        },
        "temporal_rolling_cv": {
            "folds": temporal_folds,
            "mean_roc_auc": mean_temp_roc,
            "std_roc_auc": std_temp_roc,
            "mean_pr_auc": mean_temp_pr,
            "std_pr_auc": std_temp_pr,
            "mean_precision_at_0.1pct": mean_temp_p01,
            "mean_precision_at_0.5pct": mean_temp_p05,
        },
    }

    out_file = REPORTS_DIR / "statistical_benchmarking_results.json"
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f"\n[OK] Exported full benchmark results to: {out_file}")

    # 7. Print Formatted Markdown Summary Table
    print("\n" + "=" * 80)
    print("### STATISTICAL BENCHMARKING SUMMARY TABLE")
    print("=" * 80)
    summary_md = f"""
| Model / Detection Approach | ROC-AUC (95% CI) | PR-AUC (95% CI) | Precision @ 0.1% (95% CI) | Precision @ 0.5% (95% CI) | Brier Score Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned Calibrated XGBoost (Ours)** | **{xgb_metrics['roc_auc']:.4f}** [{xgb_ci['roc_auc']['ci_lower_2.5%']:.4f}, {xgb_ci['roc_auc']['ci_upper_97.5%']:.4f}] | **{xgb_metrics['pr_auc']:.4f}** [{xgb_ci['pr_auc']['ci_lower_2.5%']:.4f}, {xgb_ci['pr_auc']['ci_upper_97.5%']:.4f}] | **{xgb_metrics['precision_at_0.1pct']:.1%}** [{xgb_ci['precision_at_0.1pct']['ci_lower_2.5%']:.1%}, {xgb_ci['precision_at_0.1pct']['ci_upper_97.5%']:.1%}] | **{xgb_metrics['precision_at_0.5pct']:.1%}** [{xgb_ci['precision_at_0.5pct']['ci_lower_2.5%']:.1%}, {xgb_ci['precision_at_0.5pct']['ci_upper_97.5%']:.1%}] | **{xgb_metrics['brier_score']:.4f}** |
| Rule: Combined AML Heuristic | {baseline_metrics['Rule: Combined AML Heuristic']['roc_auc']:.4f} [{baseline_ci['Rule: Combined AML Heuristic']['roc_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: Combined AML Heuristic']['roc_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: Combined AML Heuristic']['pr_auc']:.4f} [{baseline_ci['Rule: Combined AML Heuristic']['pr_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: Combined AML Heuristic']['pr_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: Combined AML Heuristic']['precision_at_0.1pct']:.1%} [{baseline_ci['Rule: Combined AML Heuristic']['precision_at_0.1pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: Combined AML Heuristic']['precision_at_0.1pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: Combined AML Heuristic']['precision_at_0.5pct']:.1%} [{baseline_ci['Rule: Combined AML Heuristic']['precision_at_0.5pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: Combined AML Heuristic']['precision_at_0.5pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: Combined AML Heuristic']['brier_score']:.4f} |
| Rule: High Velocity | {baseline_metrics['Rule: High Velocity']['roc_auc']:.4f} [{baseline_ci['Rule: High Velocity']['roc_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: High Velocity']['roc_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: High Velocity']['pr_auc']:.4f} [{baseline_ci['Rule: High Velocity']['pr_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: High Velocity']['pr_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: High Velocity']['precision_at_0.1pct']:.1%} [{baseline_ci['Rule: High Velocity']['precision_at_0.1pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: High Velocity']['precision_at_0.1pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: High Velocity']['precision_at_0.5pct']:.1%} [{baseline_ci['Rule: High Velocity']['precision_at_0.5pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: High Velocity']['precision_at_0.5pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: High Velocity']['brier_score']:.4f} |
| Rule: Fan-Out / Structuring | {baseline_metrics['Rule: Fan-Out / Structuring']['roc_auc']:.4f} [{baseline_ci['Rule: Fan-Out / Structuring']['roc_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: Fan-Out / Structuring']['roc_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: Fan-Out / Structuring']['pr_auc']:.4f} [{baseline_ci['Rule: Fan-Out / Structuring']['pr_auc']['ci_lower_2.5%']:.4f}, {baseline_ci['Rule: Fan-Out / Structuring']['pr_auc']['ci_upper_97.5%']:.4f}] | {baseline_metrics['Rule: Fan-Out / Structuring']['precision_at_0.1pct']:.1%} [{baseline_ci['Rule: Fan-Out / Structuring']['precision_at_0.1pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: Fan-Out / Structuring']['precision_at_0.1pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: Fan-Out / Structuring']['precision_at_0.5pct']:.1%} [{baseline_ci['Rule: Fan-Out / Structuring']['precision_at_0.5pct']['ci_lower_2.5%']:.1%}, {baseline_ci['Rule: Fan-Out / Structuring']['precision_at_0.5pct']['ci_upper_97.5%']:.1%}] | {baseline_metrics['Rule: Fan-Out / Structuring']['brier_score']:.4f} |
"""
    print(summary_md)
    print(f"Completed in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
