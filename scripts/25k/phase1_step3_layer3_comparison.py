import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError("Missing GNN dependencies. Install with: pip install torch torch-geometric") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class WalletGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, emb_dim, dropout=dropout)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index)
        logits = self.classifier(z).squeeze(-1)
        return z, logits


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


def _fit_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    model_name: str,
    params: dict | None,
) -> tuple[np.ndarray, int, float]:
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    if pos_count == 0:
        raise ValueError("Training split has zero positive labels; cannot train AML classifier.")

    if model_name == "xgb":
        if params is None:
            raise ValueError("Missing XGBoost params for training.")
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

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "lr":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown base model: {model_name}")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    y_prob = model.predict_proba(X_val)[:, 1]
    return y_prob, -1, elapsed


def _build_wallet_graph_inputs(
    train_df: pd.DataFrame,
    edge_mode: str,
    node_feature_mode: str,
    wallet_label_mode: str,
) -> tuple[list[str], dict[str, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    wallets = pd.Index(pd.concat([train_df["from"], train_df["to"]], ignore_index=True).dropna().unique())
    wallet_to_idx = {wallet: i for i, wallet in enumerate(wallets)}

    if len(wallets) == 0:
        raise ValueError("No wallets found in training split; cannot build graph for GNN.")

    edge_pairs = []
    in_degree = np.zeros(len(wallets), dtype=np.float32)
    out_degree = np.zeros(len(wallets), dtype=np.float32)

    for src_wallet, dst_wallet in train_df[["from", "to"]].itertuples(index=False):
        if src_wallet not in wallet_to_idx or dst_wallet not in wallet_to_idx:
            continue
        src = wallet_to_idx[src_wallet]
        dst = wallet_to_idx[dst_wallet]
        edge_pairs.append((src, dst))
        out_degree[src] += 1.0
        in_degree[dst] += 1.0

    if len(edge_pairs) == 0:
        raise ValueError("Training graph has zero edges; cannot train GraphSAGE.")

    edge_arr = np.array(edge_pairs, dtype=np.int64)
    if edge_mode == "bidirectional":
        edge_rev = edge_arr[:, [1, 0]]
        edge_all = np.vstack([edge_arr, edge_rev])
    elif edge_mode == "directed":
        edge_all = edge_arr
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    edge_index = torch.tensor(edge_all.T, dtype=torch.long)

    total_degree = in_degree + out_degree
    flow_ratio = np.divide(out_degree, in_degree + 1e-6)

    if node_feature_mode == "deg":
        x = np.column_stack([in_degree, out_degree, total_degree, flow_ratio]).astype(np.float32)
    elif node_feature_mode == "deg+flow":
        x = np.column_stack([in_degree, out_degree, total_degree, flow_ratio]).astype(np.float32)
    else:
        raise ValueError(f"Unknown node_feature_mode: {node_feature_mode}")

    x_nodes = torch.tensor(x, dtype=torch.float32)

    if wallet_label_mode == "any_pos":
        positive_wallets = set(
            pd.concat(
                [
                    train_df.loc[train_df["aml_label"].astype(int) == 1, "from"],
                    train_df.loc[train_df["aml_label"].astype(int) == 1, "to"],
                ],
                ignore_index=True,
            )
            .dropna()
            .tolist()
        )
    elif wallet_label_mode == "sender_pos_only":
        positive_wallets = set(train_df.loc[train_df["aml_label"].astype(int) == 1, "from"].dropna().tolist())
    else:
        raise ValueError(f"Unknown wallet_label_mode: {wallet_label_mode}")

    y_wallet = np.array([1.0 if w in positive_wallets else 0.0 for w in wallets], dtype=np.float32)
    y_wallet = torch.tensor(y_wallet, dtype=torch.float32)

    return wallets.tolist(), wallet_to_idx, x_nodes, edge_index, y_wallet


def _train_wallet_gnn(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y_wallet: torch.Tensor,
    seed: int,
    hidden_dim: int,
    emb_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    device: str,
) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        use_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        use_device = device

    model = WalletGNN(
        in_dim=x.shape[1],
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=dropout,
    ).to(use_device)

    x = x.to(use_device)
    edge_index = edge_index.to(use_device)
    y_wallet = y_wallet.to(use_device)

    pos_count = float(torch.sum(y_wallet == 1).item())
    neg_count = float(torch.sum(y_wallet == 0).item())
    if pos_count == 0:
        raise ValueError("No positive wallet labels in training graph; cannot supervise GNN.")

    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=use_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        _, logits = model(x, edge_index)
        loss = criterion(logits, y_wallet)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z, _ = model(x, edge_index)

    return z.detach().cpu()


def _attach_gnn_embeddings(
    base_X: pd.DataFrame,
    tx_df: pd.DataFrame,
    wallet_to_idx: dict[str, int],
    wallet_embeddings: np.ndarray,
) -> pd.DataFrame:
    emb_dim = wallet_embeddings.shape[1]

    def get_emb(wallet: str) -> np.ndarray:
        idx = wallet_to_idx.get(wallet)
        if idx is None:
            return np.zeros(emb_dim, dtype=np.float32)
        return wallet_embeddings[idx]

    from_matrix = np.vstack([get_emb(w) for w in tx_df["from"].tolist()]).astype(np.float32)
    to_matrix = np.vstack([get_emb(w) for w in tx_df["to"].tolist()]).astype(np.float32)

    gnn_cols = {}
    for i in range(emb_dim):
        gnn_cols[f"gnn_from_emb_{i}"] = from_matrix[:, i]
    for i in range(emb_dim):
        gnn_cols[f"gnn_to_emb_{i}"] = to_matrix[:, i]

    gnn_df = pd.DataFrame(gnn_cols, index=tx_df.index)
    return pd.concat([base_X.reset_index(drop=True), gnn_df.reset_index(drop=True)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3 (updated): base model vs base model + NTS."
    )
    parser.add_argument("--data", default="data/processed/25k/modeling_dataset_transactions_25k_multi.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="xgb", choices=["xgb", "rf", "lr"])
    parser.add_argument("--nts-mode", default="spread_abs", choices=["spread_abs", "spread_signed", "intensity"])
    parser.add_argument("--include-gnn-embeddings", action="store_true")
    parser.add_argument("--gnn-hidden-dim", type=int, default=16)
    parser.add_argument("--gnn-emb-dim", type=int, default=16)
    parser.add_argument("--gnn-epochs", type=int, default=240)
    parser.add_argument("--gnn-lr", type=float, default=0.003)
    parser.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gnn-dropout", type=float, default=0.0)
    parser.add_argument("--gnn-edge-mode", default="directed", choices=["directed", "bidirectional"])
    parser.add_argument("--gnn-node-feature-mode", default="deg", choices=["deg", "deg+flow"])
    parser.add_argument("--gnn-wallet-label-mode", default="any_pos", choices=["any_pos", "sender_pos_only"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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

    if args.include_gnn_embeddings:
        wallets, wallet_to_idx, x_nodes, edge_index, y_wallet = _build_wallet_graph_inputs(
            train_df=train_df,
            edge_mode=args.gnn_edge_mode,
            node_feature_mode=args.gnn_node_feature_mode,
            wallet_label_mode=args.gnn_wallet_label_mode,
        )

        gnn_train_start = time.perf_counter()
        wallet_embeddings = _train_wallet_gnn(
            x=x_nodes,
            edge_index=edge_index,
            y_wallet=y_wallet,
            seed=args.seed,
            hidden_dim=args.gnn_hidden_dim,
            emb_dim=args.gnn_emb_dim,
            epochs=args.gnn_epochs,
            lr=args.gnn_lr,
            weight_decay=args.gnn_weight_decay,
            dropout=args.gnn_dropout,
            device=args.device,
        ).numpy()
        gnn_train_seconds = time.perf_counter() - gnn_train_start

        X_train_base = _attach_gnn_embeddings(
            base_X=X_train_base,
            tx_df=train_df,
            wallet_to_idx=wallet_to_idx,
            wallet_embeddings=wallet_embeddings,
        )
        X_val_base = _attach_gnn_embeddings(
            base_X=X_val_base,
            tx_df=val_df,
            wallet_to_idx=wallet_to_idx,
            wallet_embeddings=wallet_embeddings,
        )
    else:
        gnn_train_seconds = 0.0

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

    base_params = baseline_params if args.base_model == "xgb" else None
    nts_model_params = nts_params if args.base_model == "xgb" else None

    scenarios = [
        (f"{args.base_model.upper()} (Baseline)", X_train_base, X_val_base, base_params),
        (f"{args.base_model.upper()}+NTS", X_train_nts, X_val_nts, nts_model_params),
    ]

    results = []
    for name, X_train, X_val, params in scenarios:
        y_prob, best_iteration, train_seconds = _fit_model(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=args.seed,
            model_name=args.base_model,
            params=params,
        )
        metrics = _evaluate_probs(y_val, y_prob)
        metrics["model"] = name
        metrics["feature_count"] = int(X_train.shape[1])
        metrics["train_seconds"] = float(train_seconds + gnn_train_seconds)
        metrics["best_iteration"] = int(best_iteration)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    print("=" * 108)
    print("LAYER 3 (UPDATED) - BASE MODEL VS BASE MODEL+NTS (VALIDATION SPLIT)")
    print("=" * 108)
    print(f"Data file: {data_path}")
    print(f"Train rows: {len(train_df)} | Positives: {int(np.sum(y_train == 1))} | Negatives: {int(np.sum(y_train == 0))}")
    print(f"Val rows: {len(val_df)} | Positives: {int(np.sum(y_val == 1))} | Negatives: {int(np.sum(y_val == 0))}")
    print(f"Seed: {args.seed} | NTS mode: {args.nts_mode} | Base model: {args.base_model}")
    print(f"Include GNN embeddings: {args.include_gnn_embeddings}")
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

    baseline_pr_auc = float(results_df.loc[results_df["model"] == f"{args.base_model.upper()} (Baseline)", "val_pr_auc"].iloc[0])
    nts_pr_auc = float(results_df.loc[results_df["model"] == f"{args.base_model.upper()}+NTS", "val_pr_auc"].iloc[0])
    winner = results_df.iloc[0]

    print(
        "Recommended winner for next step: "
        f"{winner['model']} (best val_pr_auc={winner['val_pr_auc']:.4f}, delta_vs_baseline={nts_pr_auc - baseline_pr_auc:+.4f})"
    )


if __name__ == "__main__":
    main()
