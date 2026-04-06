import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

AUGMENTED_FILE = DATA_PROCESSED_DIR / "augmented_transactions_peel.csv"
LABELS_FILE = DATA_PROCESSED_DIR / "transaction_labels_peel.csv"

MODEL_DATASET_FILE = DATA_PROCESSED_DIR / "modeling_dataset_transactions.csv"
SPLIT_SUMMARY_FILE = DATA_PROCESSED_DIR / "modeling_split_summary.csv"
PREP_METADATA_FILE = DATA_PROCESSED_DIR / "model_prep_metadata.json"


WALLET_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "flow_ratio",
    "pagerank",
    "betweenness",
]


def _resolve_user_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_wallet_features(transactions: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    graph.add_edges_from(zip(transactions["from"].astype(str), transactions["to"].astype(str)))

    in_degree = transactions.groupby("to").size().rename("in_degree")
    out_degree = transactions.groupby("from").size().rename("out_degree")

    wallets = pd.concat(
        [transactions["from"].astype(str), transactions["to"].astype(str)],
        ignore_index=True,
    ).drop_duplicates()
    features = pd.DataFrame({"wallet": wallets.astype(str)})

    features = features.merge(in_degree.rename_axis("wallet").reset_index(), on="wallet", how="left")
    features = features.merge(out_degree.rename_axis("wallet").reset_index(), on="wallet", how="left")

    features["in_degree"] = features["in_degree"].fillna(0).astype(int)
    features["out_degree"] = features["out_degree"].fillna(0).astype(int)

    total_degree = features["in_degree"] + features["out_degree"]
    features["flow_ratio"] = features["out_degree"].where(total_degree > 0, 0) / total_degree.where(total_degree > 0, 1)

    pagerank = nx.pagerank(graph) if graph.number_of_nodes() > 0 else {}
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() > 0 else {}

    features["pagerank"] = features["wallet"].map(pagerank).fillna(0.0)
    features["betweenness"] = features["wallet"].map(betweenness).fillna(0.0)

    return features.sort_values("wallet").reset_index(drop=True)


def _validate_inputs(transactions: pd.DataFrame, labels: pd.DataFrame) -> dict[str, int]:
    if "tx_hash" not in transactions.columns:
        raise ValueError("Transactions file must include tx_hash column.")
    if "tx_hash" not in labels.columns or "aml_label" not in labels.columns:
        raise ValueError("Labels file must include tx_hash and aml_label columns.")

    tx_hash_nulls = int(transactions["tx_hash"].isna().sum())
    label_hash_nulls = int(labels["tx_hash"].isna().sum())
    label_nulls = int(labels["aml_label"].isna().sum())

    if tx_hash_nulls > 0:
        raise ValueError(f"Transactions contain {tx_hash_nulls} null tx_hash values.")
    if label_hash_nulls > 0:
        raise ValueError(f"Labels contain {label_hash_nulls} null tx_hash values.")
    if label_nulls > 0:
        raise ValueError(f"Labels contain {label_nulls} null aml_label values.")

    duplicate_tx = int(transactions["tx_hash"].duplicated().sum())
    duplicate_labels = int(labels["tx_hash"].duplicated().sum())

    label_only_hashes = int((~labels["tx_hash"].astype(str).isin(transactions["tx_hash"].astype(str))).sum())

    return {
        "duplicate_tx_hashes_in_transactions": duplicate_tx,
        "duplicate_tx_hashes_in_labels": duplicate_labels,
        "orphan_label_hashes": label_only_hashes,
    }


def _assign_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    sorted_df = df.sort_values(["block_time_dt", "block_number", "tx_hash"]).reset_index(drop=True)
    n_rows = len(sorted_df)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)

    split_values = []
    for idx in range(n_rows):
        if idx < train_end:
            split_values.append("train")
        elif idx < val_end:
            split_values.append("val")
        else:
            split_values.append("test")

    sorted_df["split"] = split_values

    train_cutoff = sorted_df.iloc[train_end - 1]["block_time_dt"] if train_end > 0 else pd.NaT
    val_cutoff = sorted_df.iloc[val_end - 1]["block_time_dt"] if val_end > 0 else pd.NaT

    return sorted_df, train_cutoff, val_cutoff


def _attach_wallet_features(dataset: pd.DataFrame, wallet_features: pd.DataFrame) -> pd.DataFrame:
    from_features = wallet_features.rename(
        columns={
            "wallet": "from",
            "in_degree": "from_in_degree",
            "out_degree": "from_out_degree",
            "flow_ratio": "from_flow_ratio",
            "pagerank": "from_pagerank",
            "betweenness": "from_betweenness",
        }
    )

    to_features = wallet_features.rename(
        columns={
            "wallet": "to",
            "in_degree": "to_in_degree",
            "out_degree": "to_out_degree",
            "flow_ratio": "to_flow_ratio",
            "pagerank": "to_pagerank",
            "betweenness": "to_betweenness",
        }
    )

    merged = dataset.merge(from_features, on="from", how="left")
    merged = merged.merge(to_features, on="to", how="left")

    feature_cols = [
        "from_in_degree",
        "from_out_degree",
        "from_flow_ratio",
        "from_pagerank",
        "from_betweenness",
        "to_in_degree",
        "to_out_degree",
        "to_flow_ratio",
        "to_pagerank",
        "to_betweenness",
    ]

    for col in feature_cols:
        merged[col] = merged[col].fillna(0.0)

    merged["from_in_degree"] = merged["from_in_degree"].astype(int)
    merged["from_out_degree"] = merged["from_out_degree"].astype(int)
    merged["to_in_degree"] = merged["to_in_degree"].astype(int)
    merged["to_out_degree"] = merged["to_out_degree"].astype(int)

    return merged


def _create_modeling_table(transactions: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    label_lookup = (
        labels.groupby("tx_hash", as_index=False)
        .agg(
            aml_label=("aml_label", "max"),
            label_note=("label_note", lambda s: "peel_chain" if (s.astype(str) == "peel_chain").any() else "normal"),
        )
    )

    tx = transactions.copy()
    tx["tx_hash_instance"] = tx.groupby("tx_hash").cumcount().astype(int)
    tx["transaction_id"] = tx["tx_hash"].astype(str) + "_" + tx["tx_hash_instance"].astype(str)

    merged = tx.merge(label_lookup[["tx_hash", "aml_label", "label_note"]], on="tx_hash", how="left")

    if merged["aml_label"].isna().any():
        missing = int(merged["aml_label"].isna().sum())
        raise ValueError(f"{missing} transactions are missing labels after merge.")

    merged["aml_label"] = merged["aml_label"].astype(int)
    merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce")
    merged["block_number"] = pd.to_numeric(merged["block_number"], errors="coerce")
    merged["block_time_dt"] = pd.to_datetime(merged["block_time"], errors="coerce", utc=True)

    if merged["block_time_dt"].isna().any():
        missing_time = int(merged["block_time_dt"].isna().sum())
        raise ValueError(f"{missing_time} rows have invalid block_time values.")

    merged["hour_of_day"] = merged["block_time_dt"].dt.hour
    merged["day_of_week"] = merged["block_time_dt"].dt.dayofweek
    merged["amount_log1p"] = np.log1p(merged["amount"].clip(lower=0))

    split_df, train_cutoff, val_cutoff = _assign_time_split(merged)

    train_rows = split_df[split_df["split"] == "train"].copy()
    wallet_features = _build_wallet_features(train_rows[["from", "to"]].copy())

    modeling_df = _attach_wallet_features(split_df, wallet_features)

    keep_cols = [
        "transaction_id",
        "tx_hash",
        "aml_label",
        "label_note",
        "split",
        "amount",
        "amount_log1p",
        "block_number",
        "block_time",
        "hour_of_day",
        "day_of_week",
        "from",
        "to",
        "from_in_degree",
        "from_out_degree",
        "from_flow_ratio",
        "from_pagerank",
        "from_betweenness",
        "to_in_degree",
        "to_out_degree",
        "to_flow_ratio",
        "to_pagerank",
        "to_betweenness",
    ]

    modeling_df = modeling_df[keep_cols].copy()
    modeling_df = modeling_df.sort_values(["block_number", "tx_hash"]).reset_index(drop=True)

    metadata = {
        "train_time_cutoff_utc": str(train_cutoff),
        "val_time_cutoff_utc": str(val_cutoff),
        "row_count": int(len(modeling_df)),
        "label_0_count": int((modeling_df["aml_label"] == 0).sum()),
        "label_1_count": int((modeling_df["aml_label"] == 1).sum()),
    }

    return modeling_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare transaction-level AML modeling dataset with leakage-safe time split.")
    parser.add_argument("--transactions", default="data/processed/augmented_transactions_peel.csv")
    parser.add_argument("--labels", default="data/processed/transaction_labels_peel.csv")
    parser.add_argument("--output", default="data/processed/modeling_dataset_transactions.csv")
    parser.add_argument("--summary", default="data/processed/modeling_split_summary.csv")
    parser.add_argument("--metadata", default="data/processed/model_prep_metadata.json")
    args = parser.parse_args()

    transactions_path = _resolve_user_path(args.transactions)
    labels_path = _resolve_user_path(args.labels)
    output_path = _resolve_user_path(args.output)
    summary_path = _resolve_user_path(args.summary)
    metadata_path = _resolve_user_path(args.metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_csv(transactions_path)
    labels = pd.read_csv(labels_path)

    validation_stats = _validate_inputs(transactions, labels)
    modeling_df, metadata = _create_modeling_table(transactions, labels)

    split_summary = (
        modeling_df.groupby(["split", "aml_label"], as_index=False)
        .size()
        .pivot(index="split", columns="aml_label", values="size")
        .fillna(0)
        .reset_index()
        .rename(columns={0: "label_0_count", 1: "label_1_count"})
    )

    for col in ["label_0_count", "label_1_count"]:
        if col not in split_summary.columns:
            split_summary[col] = 0
        split_summary[col] = split_summary[col].astype(int)

    split_summary["total_count"] = split_summary["label_0_count"] + split_summary["label_1_count"]
    split_summary["positive_rate"] = split_summary["label_1_count"] / split_summary["total_count"].where(
        split_summary["total_count"] > 0, 1
    )

    metadata.update(validation_stats)
    metadata["splits"] = split_summary.to_dict(orient="records")

    modeling_df.to_csv(output_path, index=False)
    split_summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model-prep completed.")
    print(f"Saved modeling dataset to {output_path}")
    print(f"Saved split summary to {summary_path}")
    print(f"Saved prep metadata to {metadata_path}")
    print(f"Rows: {len(modeling_df)}")
    print(f"Label 0: {(modeling_df['aml_label'] == 0).sum()}")
    print(f"Label 1: {(modeling_df['aml_label'] == 1).sum()}")


if __name__ == "__main__":
    main()
