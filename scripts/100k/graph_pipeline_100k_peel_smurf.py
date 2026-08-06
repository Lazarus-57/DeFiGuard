"""100k peel-chain + smurfing graph pipeline.

Generates synthetic transactions, merges them with real 100k
Ethereum data, and writes augmented CSVs + wallet features to
``data/processed/100k/peel_smurf/``.
"""

import argparse
import random
from math import ceil
from pathlib import Path

import pandas as pd
import networkx as nx

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name.lower() == "scripts" else SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "scripts" / "pipeline"))

from pattern_injection import generate_peel_chains, generate_smurf_clusters

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "100k"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "100k" / "peel_smurf"

SOURCE_FILE = DATA_RAW_DIR / "eth_transfers_100k_2024.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    expected_tx = _expected_transactions_per_chain(chain_length_weights)
    target_pos = (target_positive_rate * n_base_rows) / (1.0 - target_positive_rate)
    return max(1, int(ceil(target_pos / expected_tx)))


def _expected_transactions_per_smurf(mule_count_weights: dict[int, float]) -> float:
    total_weight = float(sum(mule_count_weights.values()))
    if total_weight <= 0:
        raise ValueError("mule_count_weights must have a positive total weight.")
    expected = 0.0
    for mule_count, weight in mule_count_weights.items():
        expected += (weight / total_weight) * (2 * mule_count)
    return expected


def _estimate_smurf_count_for_target_rate(
    n_base_rows: int,
    target_positive_rate: float,
    mule_count_weights: dict[int, float],
) -> int:
    if not (0.0 < target_positive_rate < 1.0):
        raise ValueError("target_positive_rate must be between 0 and 1.")
    expected_tx = _expected_transactions_per_smurf(mule_count_weights)
    target_pos = (target_positive_rate * n_base_rows) / (1.0 - target_positive_rate)
    return max(1, int(ceil(target_pos / expected_tx)))


def build_wallet_features(transactions: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    graph.add_edges_from(zip(transactions["from"], transactions["to"]))

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph)
    betweenness = nx.betweenness_centrality(graph, k=min(500, graph.number_of_nodes()), seed=42)

    wallets = list(graph.nodes())
    features = pd.DataFrame({"wallet": wallets})
    features["in_degree"] = features["wallet"].map(in_degree).fillna(0).astype(int)
    features["out_degree"] = features["wallet"].map(out_degree).fillna(0).astype(int)

    total_degree = features["in_degree"] + features["out_degree"]
    features["flow_ratio"] = (
        features["out_degree"].where(total_degree > 0, 0)
        / total_degree.where(total_degree > 0, 1)
    )

    features["pagerank"] = features["wallet"].map(pagerank).fillna(0.0)
    features["betweenness"] = features["wallet"].map(betweenness).fillna(0.0)

    return features.sort_values("wallet").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build augmented 100k AML dataset with peel-chain AND smurfing patterns."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--peel-target-rate",
        type=float,
        default=0.04,
        help="Target peel-chain positive rate (default: 0.04).",
    )
    parser.add_argument(
        "--smurf-target-rate",
        type=float,
        default=0.03,
        help="Target smurfing positive rate (default: 0.03).",
    )
    parser.add_argument("--num-peel-chains", type=int, default=None)
    parser.add_argument("--num-smurf-clusters", type=int, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_file = Path(args.source) if args.source else SOURCE_FILE
    if not source_file.is_absolute():
        source_file = (PROJECT_ROOT / source_file).resolve()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_PROCESSED_DIR
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    augmented_file = output_dir / "augmented_transactions.csv"
    features_file = output_dir / "wallet_features.csv"
    labels_file = output_dir / "transaction_labels.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load base data ----
    df = pd.read_csv(source_file)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0) / 1e18

    # ---- Shared context for synthetic generation ----
    existing_wallets = set(df["from"].dropna().astype(str)).union(set(df["to"].dropna().astype(str)))
    real_amounts_raw = df["amount"].dropna().tolist()
    real_amounts = [a for a in real_amounts_raw if a > 0]
    anchor_wallets = list(set(df["from"].dropna().astype(str).tolist()))

    # ---- Peel chains ----
    chain_length_weights = {4: 0.2, 5: 0.5, 6: 0.3}
    num_peel = args.num_peel_chains
    if num_peel is None:
        num_peel = _estimate_chain_count_for_target_rate(
            n_base_rows=len(df),
            target_positive_rate=args.peel_target_rate,
            chain_length_weights=chain_length_weights,
        )

    peel_df = generate_peel_chains(
        num_chains=num_peel,
        existing_wallets=existing_wallets,
        chain_length_weights=chain_length_weights,
        seed=args.seed,
        real_amounts=real_amounts if real_amounts else None,
        anchor_wallets=anchor_wallets if anchor_wallets else None,
    )

    peel_wallets = set(peel_df["from"].tolist()) | set(peel_df["to"].tolist())
    existing_wallets = existing_wallets | peel_wallets

    # ---- Smurfing clusters ----
    mule_count_weights = {3: 0.1, 4: 0.2, 5: 0.3, 6: 0.25, 7: 0.1, 8: 0.05}
    num_smurf = args.num_smurf_clusters
    if num_smurf is None:
        num_smurf = _estimate_smurf_count_for_target_rate(
            n_base_rows=len(df),
            target_positive_rate=args.smurf_target_rate,
            mule_count_weights=mule_count_weights,
        )

    smurf_df = generate_smurf_clusters(
        num_clusters=num_smurf,
        existing_wallets=existing_wallets,
        mule_count_weights=mule_count_weights,
        seed=args.seed + 1000,
        real_amounts=real_amounts if real_amounts else None,
        anchor_wallets=anchor_wallets if anchor_wallets else None,
    )

    # ---- Assign timestamps and block numbers to synthetic rows ----
    parsed_time = pd.to_datetime(df.get("block_time"), errors="coerce", utc=True)
    valid_times = [t for t in parsed_time.dropna().tolist()]
    if not valid_times:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        valid_times = [now_utc]

    block_numbers = pd.to_numeric(df.get("block_number"), errors="coerce").dropna().astype(int).tolist()
    if not block_numbers:
        block_numbers = [0]

    def _assign_sequential_times_peel(df: pd.DataFrame) -> tuple[list[str], list[int]]:
        times, blocks = [], []
        current_time_map = {}
        for idx, row in df.iterrows():
            pid = row["pattern_id"]
            if pid not in current_time_map:
                current_time_map[pid] = rng.choice(valid_times)
            
            current_time_map[pid] += pd.Timedelta(seconds=rng.randint(60, 300))
            times.append(current_time_map[pid].strftime("%Y-%m-%d %H:%M:%S.000 UTC"))
            blocks.append(rng.choice(block_numbers))
        return times, blocks

    def _assign_sequential_times_smurf(df: pd.DataFrame) -> tuple[list[str], list[int]]:
        times, blocks = [], []
        current_time_map = {}
        source_map = {}
        
        for idx, row in df.iterrows():
            pid = row["pattern_id"]
            if pid not in current_time_map:
                current_time_map[pid] = rng.choice(valid_times)
                source_map[pid] = row["from"]
            
            if row["from"] == source_map[pid]:
                current_time_map[pid] += pd.Timedelta(seconds=rng.randint(1, 30))
            else:
                if source_map[pid] is not None:
                    current_time_map[pid] += pd.Timedelta(seconds=rng.randint(300, 900))
                    source_map[pid] = None
                else:
                    current_time_map[pid] += pd.Timedelta(seconds=rng.randint(1, 30))

            times.append(current_time_map[pid].strftime("%Y-%m-%d %H:%M:%S.000 UTC"))
            blocks.append(rng.choice(block_numbers))
        return times, blocks

    peel_times, peel_blocks = _assign_sequential_times_peel(peel_df)
    smurf_times, smurf_blocks = _assign_sequential_times_smurf(smurf_df)

    peel_synthetic = pd.DataFrame({
        "amount": peel_df["amount"],
        "block_number": peel_blocks,
        "block_time": peel_times,
        "from": peel_df["from"],
        "to": peel_df["to"],
        "tx_hash": [f"0x{rng.getrandbits(256):064x}" for _ in range(len(peel_df))],
    })

    smurf_synthetic = pd.DataFrame({
        "amount": smurf_df["amount"],
        "block_number": smurf_blocks,
        "block_time": smurf_times,
        "from": smurf_df["from"],
        "to": smurf_df["to"],
        "tx_hash": [f"0x{rng.getrandbits(256):064x}" for _ in range(len(smurf_df))],
    })

    # ---- Combine everything ----
    combined = pd.concat([df, peel_synthetic, smurf_synthetic], ignore_index=True)
    combined.to_csv(augmented_file, index=False)

    # ---- Labels ----
    labels_normal = pd.DataFrame({
        "tx_hash": df["tx_hash"],
        "aml_label": 0,
        "label_note": "normal",
    })
    labels_peel = pd.DataFrame({
        "tx_hash": peel_synthetic["tx_hash"],
        "aml_label": 1,
        "label_note": "peel_chain",
    })
    labels_smurf = pd.DataFrame({
        "tx_hash": smurf_synthetic["tx_hash"],
        "aml_label": 1,
        "label_note": "smurfing",
    })
    labels = pd.concat([labels_normal, labels_peel, labels_smurf], ignore_index=True)
    labels.to_csv(labels_file, index=False)

    # ---- Wallet features ----
    feature_input = combined[["from", "to", "amount"]].copy()
    features = build_wallet_features(feature_input)
    features.to_csv(features_file, index=False)

    # ---- Summary ----
    print(f"Source file: {source_file}")
    print(f"Output directory: {output_dir}")
    print(f"Original transactions: {len(df)}")
    print()
    print(f"Peel-chain target rate: {args.peel_target_rate:.4f}")
    print(f"Peel-chain count: {num_peel}")
    print(f"Peel-chain transactions: {len(peel_df)}")
    print()
    print(f"Smurfing target rate: {args.smurf_target_rate:.4f}")
    print(f"Smurfing cluster count: {num_smurf}")
    print(f"Smurfing transactions: {len(smurf_df)}")
    print()
    print(f"Total augmented transactions: {len(combined)}")
    total_pos = len(peel_df) + len(smurf_df)
    print(f"Total positives: {total_pos}")
    print(f"Actual positive rate: {total_pos / max(len(combined), 1):.4f}")
    print()
    print(f"Saved augmented dataset to {augmented_file}")
    print(f"Saved label mapping to {labels_file}")
    print(f"Saved wallet features to {features_file}")
    print(f"Seed used: {args.seed}")


if __name__ == "__main__":
    main()
