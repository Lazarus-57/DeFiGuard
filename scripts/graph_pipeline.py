import argparse
import pandas as pd
import networkx as nx
import random
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
    args = parser.parse_args()

    rng = random.Random(args.seed)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_FILE)

    existing_wallets = set(df["from"].dropna().astype(str)).union(set(df["to"].dropna().astype(str)))
    peel_df = generate_peel_chains(
        num_chains=50,
        existing_wallets=existing_wallets,
        chain_length_weights={4: 0.2, 5: 0.5, 6: 0.3},
        seed=args.seed,
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
    combined.to_csv(AUGMENTED_FILE, index=False)

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
    labels.to_csv(LABELS_FILE, index=False)

    feature_input = combined[["from", "to", "amount"]].copy()
    features = build_wallet_features(feature_input)
    features.to_csv(FEATURES_FILE, index=False)

    print(f"Original transactions: {len(df)}")
    print(f"Synthetic peel-chain transactions: {len(peel_df)}")
    print(f"Augmented transactions: {len(combined)}")
    print(f"Saved augmented dataset to {AUGMENTED_FILE}")
    print(f"Saved label mapping to {LABELS_FILE}")
    print(f"Saved wallet features to {FEATURES_FILE}")
    print(f"Seed used: {args.seed}")


if __name__ == "__main__":
    main()
