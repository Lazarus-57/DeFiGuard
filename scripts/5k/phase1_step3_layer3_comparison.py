import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def _resolve_user_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_minmax(series: pd.Series) -> pd.Series:
    vmin = float(series.min())
    vmax = float(series.max())
    if np.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - vmin) / (vmax - vmin)


def _compute_nts_map(train_df: pd.DataFrame, mode: str = "spread_abs") -> dict[str, float]:
    local = train_df[["from", "to", "block_time"]].copy()
    local["block_time"] = pd.to_datetime(local["block_time"], utc=True, errors="coerce")
    local = local.dropna(subset=["from", "to", "block_time"])

    in_stats = local.groupby("to")["block_time"].agg(["min", "max", "count"])
    out_stats = local.groupby("from")["block_time"].agg(["min", "max", "count"])

    in_spread = (in_stats["max"] - in_stats["min"]).dt.total_seconds()
    out_spread = (out_stats["max"] - out_stats["min"]).dt.total_seconds()

    wallets = pd.Index(in_spread.index).union(out_spread.index)
    in_spread = in_spread.reindex(wallets).fillna(0.0)
    out_spread = out_spread.reindex(wallets).fillna(0.0)

    if mode == "spread_abs":
        theta = (out_spread - in_spread).abs()
    elif mode == "spread_signed":
        theta = out_spread - in_spread
    elif mode == "intensity":
        in_cnt = in_stats["count"].reindex(wallets).fillna(0.0)
        out_cnt = out_stats["count"].reindex(wallets).fillna(0.0)
        theta = (out_cnt - in_cnt).abs() / (out_cnt + in_cnt + 1e-9)
    else:
        raise ValueError(f"Unknown nts mode: {mode}")

    nts = _safe_minmax(theta)
    return nts.to_dict()


def _attach_nts_features(tx_df: pd.DataFrame, base_X: pd.DataFrame, nts_map: dict[str, float]) -> pd.DataFrame:
    from_wallet = tx_df["from"].fillna("")
    to_wallet = tx_df["to"].fillna("")

    from_nts = from_wallet.map(nts_map).fillna(0.0).astype(float)
    to_nts = to_wallet.map(nts_map).fillna(0.0).astype(float)

    nts_df = pd.DataFrame(
        {
            "from_nts": from_nts,
            "to_nts": to_nts,
            "nts_max": np.maximum(from_nts, to_nts),
            "nts_mean": (from_nts + to_nts) / 2.0,
        },
        index=tx_df.index,
    )

    return pd.concat([base_X.reset_index(drop=True), nts_df.reset_index(drop=True)], axis=1)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    num = 2 * p * r
    f1 = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom > 0)
    return float(thresholds[int(np.nanargmax(f1))])


def _recall_at_precision_target(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.80) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid_recalls = recall[precision >= target_precision]
    if len(valid_recalls) == 0:
        return 0.0
    return float(np.max(valid_recalls))


def _evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    threshold = _best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "val_pr_auc": float(average_precision_score(y_true, y_prob)),
        "val_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "val_recall_at_p80": float(_recall_at_precision_target(y_true, y_prob, target_precision=0.80)),
        "threshold": float(threshold),
    }


def _fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    params: dict,
) -> tuple[np.ndarray, int, float]:
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    if pos_count == 0:
        raise ValueError("Training split has zero positive labels; cannot train AML classifier.")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        booster="gbtree",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=neg_count / pos_count,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.perf_counter() - start

    y_prob = model.predict_proba(X_val)[:, 1]
    best_iteration = int(getattr(model, "best_iteration", -1))
    return y_prob, best_iteration, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3 (updated): tuned standalone XGBoost vs tuned XGBoost+NTS."
    )
    parser.add_argument("--data", default="data/processed/5k/modeling_dataset_transactions.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nts-mode", default="spread_abs", choices=["spread_abs", "spread_signed", "intensity"])
    args = parser.parse_args()

    data_path = _resolve_user_path(args.data)
    df = pd.read_csv(data_path)

    required_cols = {"split", "aml_label", "from", "to", "block_time"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input dataset missing required columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty in the provided dataset.")

    y_train = train_df["aml_label"].astype(int).to_numpy()
    y_val = val_df["aml_label"].astype(int).to_numpy()

    X_train_base = _prepare_base_features(train_df)
    X_val_base = _prepare_base_features(val_df)

    nts_map = _compute_nts_map(train_df, mode=args.nts_mode)
    X_train_nts = _attach_nts_features(train_df, X_train_base, nts_map)
    X_val_nts = _attach_nts_features(val_df, X_val_base, nts_map)

    # Best baseline and best NTS-oriented settings from the sweep runs.
    baseline_params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
    }
    nts_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    }

    scenarios = [
        ("XGBoost (Baseline tuned)", X_train_base, X_val_base, baseline_params),
        ("XGBoost+NTS (Layer3 tuned)", X_train_nts, X_val_nts, nts_params),
    ]

    results = []
    for name, X_train, X_val, params in scenarios:
        y_prob, best_iteration, train_seconds = _fit_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=args.seed,
            params=params,
        )
        metrics = _evaluate_probs(y_val, y_prob)
        metrics["model"] = name
        metrics["feature_count"] = int(X_train.shape[1])
        metrics["train_seconds"] = float(train_seconds)
        metrics["best_iteration"] = int(best_iteration)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    print("=" * 108)
    print("LAYER 3 (UPDATED) - TUNED XGBOOST VS TUNED XGBOOST+NTS (VALIDATION SPLIT)")
    print("=" * 108)
    print(f"Data file: {data_path}")
    print(f"Train rows: {len(train_df)} | Positives: {int(np.sum(y_train == 1))} | Negatives: {int(np.sum(y_train == 0))}")
    print(f"Val rows: {len(val_df)} | Positives: {int(np.sum(y_val == 1))} | Negatives: {int(np.sum(y_val == 0))}")
    print(f"Seed: {args.seed} | NTS mode: {args.nts_mode}")
    print("NTS is computed from TRAIN split only and mapped to VAL.")
    print("-" * 108)

    display_cols = [
        "model",
        "val_pr_auc",
        "val_roc_auc",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_recall_at_p80",
        "threshold",
        "feature_count",
        "train_seconds",
        "best_iteration",
    ]

    printable = results_df[display_cols].copy()
    float_cols = [c for c in printable.columns if c not in {"model", "best_iteration", "feature_count"}]
    for col in float_cols:
        printable[col] = printable[col].astype(float).round(4)

    print(printable.to_string(index=False))
    print("-" * 108)

    baseline_pr_auc = float(results_df.loc[results_df["model"] == "XGBoost (Baseline tuned)", "val_pr_auc"].iloc[0])
    nts_pr_auc = float(results_df.loc[results_df["model"] == "XGBoost+NTS (Layer3 tuned)", "val_pr_auc"].iloc[0])
    winner = results_df.iloc[0]

    print(
        "Recommended winner for next step: "
        f"{winner['model']} (best val_pr_auc={winner['val_pr_auc']:.4f}, delta_vs_baseline={nts_pr_auc - baseline_pr_auc:+.4f})"
    )


if __name__ == "__main__":
    main()
