import argparse
import pandas as pd
import networkx as nx
import random
from math import ceil
from pathlib import Path

from pattern_injection import generate_peel_chains


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCE_FILE = DATA_RAW_DIR / "eth_transfers_2024_01_01_1hr.csv"
AUGMENTED_FILE = DATA_PROCESSED_DIR / "augmented_transactions_peel.csv"
FEATURES_FILE = DATA_PROCESSED_DIR / "wallet_features.csv"
LABELS_FILE = DATA_PROCESSED_DIR / "transaction_labels_peel.csv"


def _expected_transactions_per_chain(chain_length_weights: dict[int, float]) -> float:
    total_weight = float(sum(chain_length_weights.values()))
    if total_weight <= 0:
        raise ValueError("chain_length_weights must have a positive total weight.")

    expected = 0.0
    for chain_length, weight in chain_length_weights.items():
        if chain_length < 2:
            raise ValueError("Each chain length must be >= 2.")
        expected += (weight / total_weight) * (chain_length - 1)
    return expected


def _estimate_chain_count_for_target_rate(
    n_base_rows: int,
    target_positive_rate: float,
    chain_length_weights: dict[int, float],
) -> int:
    if not (0.0 < target_positive_rate < 1.0):
        raise ValueError("target_positive_rate must be between 0 and 1.")

    expected_tx_per_chain = _expected_transactions_per_chain(chain_length_weights)
    target_positive_count = (target_positive_rate * n_base_rows) / (1.0 - target_positive_rate)
    return max(1, int(ceil(target_positive_count / expected_tx_per_chain)))


def build_wallet_features(transactions: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    graph.add_edges_from(zip(transactions["from"], transactions["to"]))

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph)
    betweenness = nx.betweenness_centrality(graph)

    wallets = list(graph.nodes())
    features = pd.DataFrame({"wallet": wallets})
    features["in_degree"] = features["wallet"].map(in_degree).fillna(0).astype(int)
    features["out_degree"] = features["wallet"].map(out_degree).fillna(0).astype(int)

    total_degree = features["in_degree"] + features["out_degree"]
    features["flow_ratio"] = (
        features["out_degree"].where(total_degree > 0, 0) / total_degree.where(total_degree > 0, 1)
    )

    features["pagerank"] = features["wallet"].map(pagerank).fillna(0.0)
    features["betweenness"] = features["wallet"].map(betweenness).fillna(0.0)

    return features.sort_values("wallet").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build augmented AML dataset with reproducible synthetic peel chains.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic synthetic generation.")
    parser.add_argument(
        "--target-positive-rate",
        type=float,
        default=0.04,
        help="Target positive rate in augmented dataset (default: 0.04).",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=None,
        help="Optional fixed number of peel chains. If omitted, computed from target-positive-rate.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to raw source CSV. Defaults to data/raw/eth_transfers_2024_01_01_1hr.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed files. Defaults to data/processed/.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_file = Path(args.source) if args.source else SOURCE_FILE
    if not source_file.is_absolute():
        source_file = (PROJECT_ROOT / source_file).resolve()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_PROCESSED_DIR
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    augmented_file = output_dir / "augmented_transactions_peel.csv"
    features_file = output_dir / "wallet_features.csv"
    labels_file = output_dir / "transaction_labels_peel.csv"

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_file)

    # Convert raw Wei amounts to ETH for numeric consistency with synthetic data.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0) / 1e18

    chain_length_weights = {4: 0.2, 5: 0.5, 6: 0.3}
    num_chains = args.num_chains
    if num_chains is None:
        num_chains = _estimate_chain_count_for_target_rate(
            n_base_rows=len(df),
            target_positive_rate=args.target_positive_rate,
            chain_length_weights=chain_length_weights,
        )

    existing_wallets = set(df["from"].dropna().astype(str)).union(set(df["to"].dropna().astype(str)))

    # Collect real amounts (non-zero, in ETH) and real wallet addresses
    # so synthetic peel chains blend into the real data distribution.
    real_amounts_raw = df["amount"].dropna().tolist()
    real_amounts = [a for a in real_amounts_raw if a > 0]
    anchor_wallets = list(set(df["from"].dropna().astype(str).tolist()))

    peel_df = generate_peel_chains(
        num_chains=num_chains,
        existing_wallets=existing_wallets,
        chain_length_weights=chain_length_weights,
        seed=args.seed,
        real_amounts=real_amounts if real_amounts else None,
        anchor_wallets=anchor_wallets if anchor_wallets else None,
    )

    parsed_time = pd.to_datetime(df.get("block_time"), errors="coerce", utc=True)
    valid_times = [t for t in parsed_time.dropna().tolist()]
    if not valid_times:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        valid_times = [now_utc]

    block_numbers = pd.to_numeric(df.get("block_number"), errors="coerce").dropna().astype(int).tolist()
    if not block_numbers:
        block_numbers = [0]

    sampled_times = []
    sampled_blocks = []
    for _ in range(len(peel_df)):
        base_time = rng.choice(valid_times)
        offset_seconds = rng.randint(0, 59)
        sampled_times.append((base_time + pd.Timedelta(seconds=offset_seconds)).strftime("%Y-%m-%d %H:%M:%S.000 UTC"))
        sampled_blocks.append(rng.choice(block_numbers))

    synthetic = pd.DataFrame(
        {
            "amount": peel_df["amount"],
            "block_number": sampled_blocks,
            "block_time": sampled_times,
            "from": peel_df["from"],
            "to": peel_df["to"],
            "tx_hash": [f"0x{rng.getrandbits(256):064x}" for _ in range(len(peel_df))],
        }
    )

    df_augmented = df.copy()

    combined = pd.concat([df_augmented, synthetic], ignore_index=True)
    combined.to_csv(augmented_file, index=False)

    labels_normal = pd.DataFrame(
        {
            "tx_hash": df_augmented["tx_hash"],
            "aml_label": 0,
            "label_note": "normal",
        }
    )
    labels_peel = pd.DataFrame(
        {
            "tx_hash": synthetic["tx_hash"],
            "aml_label": 1,
            "label_note": "peel_chain",
        }
    )
    labels = pd.concat([labels_normal, labels_peel], ignore_index=True)
    labels.to_csv(labels_file, index=False)

    feature_input = combined[["from", "to", "amount"]].copy()
    features = build_wallet_features(feature_input)
    features.to_csv(features_file, index=False)

    print(f"Source file: {source_file}")
    print(f"Output directory: {output_dir}")
    print(f"Original transactions: {len(df)}")
    print(f"Configured target positive rate: {args.target_positive_rate:.4f}")
    print(f"Configured peel-chain count: {num_chains}")
    print(f"Synthetic peel-chain transactions: {len(peel_df)}")
    print(f"Augmented transactions: {len(combined)}")
    print(f"Actual positive rate: {len(peel_df) / max(len(combined), 1):.4f}")
    print(f"Saved augmented dataset to {augmented_file}")
    print(f"Saved label mapping to {labels_file}")
    print(f"Saved wallet features to {features_file}")
    print(f"Seed used: {args.seed}")


if __name__ == "__main__":
    main()
