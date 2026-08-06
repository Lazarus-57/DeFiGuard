import argparse
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = SCRIPT_DIR.parent / "Phase 1 Models"
REPORTS_DIR = SCRIPT_DIR.parent / "Phase 1 Reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --- GNN Definitions ---
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


def _build_wallet_graph_inputs(train_df: pd.DataFrame):
    wallets = pd.Index(pd.concat([train_df["from"], train_df["to"]], ignore_index=True).dropna().unique())
    wallet_to_idx = {wallet: i for i, wallet in enumerate(wallets)}

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

    edge_arr = np.array(edge_pairs, dtype=np.int64)
    edge_index = torch.tensor(edge_arr.T, dtype=torch.long)

    total_degree = in_degree + out_degree
    flow_ratio = np.divide(out_degree, in_degree + 1e-6)
    x = np.column_stack([in_degree, out_degree, total_degree, flow_ratio]).astype(np.float32)
    x_nodes = torch.tensor(x, dtype=torch.float32)

    positive_wallets = set(
        pd.concat(
            [
                train_df.loc[train_df["aml_label"].astype(int) == 1, "from"],
                train_df.loc[train_df["aml_label"].astype(int) == 1, "to"],
            ],
            ignore_index=True,
        ).dropna().tolist()
    )

    y_wallet = np.array([1.0 if w in positive_wallets else 0.0 for w in wallets], dtype=np.float32)
    y_wallet = torch.tensor(y_wallet, dtype=torch.float32)

    return wallets.tolist(), wallet_to_idx, x_nodes, edge_index, y_wallet


def _train_wallet_gnn(x, edge_index, y_wallet, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WalletGNN(in_dim=x.shape[1], hidden_dim=16, emb_dim=16, dropout=0.0).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y_wallet = y_wallet.to(device)

    pos_count = float(torch.sum(y_wallet == 1).item())
    neg_count = float(torch.sum(y_wallet == 0).item())
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    model.train()
    for _ in range(240):
        optimizer.zero_grad()
        _, logits = model(x, edge_index)
        loss = criterion(logits, y_wallet)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z, _ = model(x, edge_index)

    return z.detach().cpu().numpy()


def _attach_gnn_embeddings(base_X, tx_df, wallet_to_idx, wallet_embeddings):
    emb_dim = wallet_embeddings.shape[1]

    def get_emb(wallet):
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


# --- NTS Definitions ---
def _safe_minmax(series: pd.Series) -> pd.Series:
    vmin = float(series.min())
    vmax = float(series.max())
    if np.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - vmin) / (vmax - vmin)

def _compute_nts_map(train_df: pd.DataFrame) -> dict:
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

    theta = (out_spread - in_spread).abs()
    nts = _safe_minmax(theta)
    return nts.to_dict()

def _attach_nts_features(tx_df: pd.DataFrame, base_X: pd.DataFrame, nts_map: dict) -> pd.DataFrame:
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


# --- Core Pipeline ---
def _prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {"transaction_id", "tx_hash", "aml_label", "label_note", "split", "block_time", "from", "to"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    num = 2 * p * r
    f1 = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom > 0)
    return float(thresholds[int(np.nanargmax(f1))])


def main():
    print("Loading datasets...")
    data_path = PROJECT_ROOT / "data" / "processed" / "100k" / "all_patterns" / "modeling_dataset.csv"
    df = pd.read_csv(data_path)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df["aml_label"].astype(int).to_numpy()
    y_val = val_df["aml_label"].astype(int).to_numpy()
    y_test = test_df["aml_label"].astype(int).to_numpy()

    X_train_base = _prepare_base_features(train_df)
    X_val_base = _prepare_base_features(val_df)
    X_test_base = _prepare_base_features(test_df)

    print("1. Computing GNN Embeddings on Train Split...")
    wallets, wallet_to_idx, x_nodes, edge_index, y_wallet = _build_wallet_graph_inputs(train_df)
    wallet_embeddings = _train_wallet_gnn(x_nodes, edge_index, y_wallet)
    
    print("2. Mapping GNN Embeddings...")
    X_train = _attach_gnn_embeddings(X_train_base, train_df, wallet_to_idx, wallet_embeddings)
    X_val = _attach_gnn_embeddings(X_val_base, val_df, wallet_to_idx, wallet_embeddings)
    X_test = _attach_gnn_embeddings(X_test_base, test_df, wallet_to_idx, wallet_embeddings)

    print("3. Computing NTS Map on Train Split...")
    nts_map = _compute_nts_map(train_df)
    
    print("4. Mapping NTS Features...")
    X_train = _attach_nts_features(train_df, X_train, nts_map)
    X_val = _attach_nts_features(val_df, X_val, nts_map)
    X_test = _attach_nts_features(test_df, X_test, nts_map)

    feature_names = list(X_train.columns)

    print(f"5. Training Master Hybrid XGBoost Model ({len(feature_names)} features)...")
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        booster="gbtree",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=1.0,
        scale_pos_weight=neg_count / pos_count,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    print("6. Evaluating on Blind Test Set...")
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # We use validation set to pick the best F1 threshold, and apply it to test
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold = _best_f1_threshold(y_val, y_prob_val)
    
    y_pred_test = (y_prob_test >= threshold).astype(int)

    metrics = {
        "test_pr_auc": float(average_precision_score(y_test, y_prob_test)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "decision_threshold": threshold
    }
    
    print("Test Metrics:", metrics)
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("7. Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(REPORTS_DIR / "shap_summary.png", bbox_inches='tight', dpi=300)
    plt.close()

    print("8. Serializing Model Artifacts...")
    model.save_model(MODELS_DIR / "master_hybrid_model.json")
    
    with open(MODELS_DIR / "gnn_wallet_embeddings.pkl", "wb") as f:
        pickle.dump({"wallet_to_idx": wallet_to_idx, "embeddings": wallet_embeddings}, f)
        
    with open(MODELS_DIR / "nts_map.pkl", "wb") as f:
        pickle.dump(nts_map, f)
        
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)
        
    with open(MODELS_DIR / "decision_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=4)
        
    with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    print("Phase 1 Master Training Complete.")


if __name__ == "__main__":
    main()
