import pandas as pd
import numpy as np
import sys

pipeline = sys.argv[1] if len(sys.argv) > 1 else "5k"

PIPELINE_PATHS = {
    "5k": "data/processed/5k/modeling_dataset_transactions.csv",
    "25k": "data/processed/25k/modeling_dataset_transactions_25k.csv",
    "25k-multi": "data/processed/25k/modeling_dataset_transactions_25k_multi.csv",
}

if pipeline not in PIPELINE_PATHS:
    print(f"Unknown pipeline '{pipeline}'. Available: {', '.join(PIPELINE_PATHS.keys())}")
    sys.exit(1)

data_path = PIPELINE_PATHS[pipeline]
df = pd.read_csv(data_path)
print(f"=== {pipeline.upper()} DATASET OVERVIEW ===")
print(f"Total rows: {len(df)}")
print(f"Raw CSV file bytes: N/A (run separately)")
print()

positives = df[df["aml_label"] == 1]
negatives = df[df["aml_label"] == 0]

feat_cols = [
    "amount", "from_in_degree", "from_out_degree",
    "from_pagerank", "from_betweenness",
    "to_in_degree", "to_out_degree",
    "to_pagerank", "to_betweenness"
]

print("=== FEATURE STATS: POSITIVES (all patterns) ===")
print(positives[feat_cols].describe().round(6).to_string())
print()
print("=== FEATURE STATS: NEGATIVES (normal) ===")
print(negatives[feat_cols].describe().round(6).to_string())
print()

# Per-pattern breakdown (only for multi-pattern datasets)
if "label_note" in df.columns:
    pattern_counts = df["label_note"].value_counts()
    print("=== LABEL NOTE DISTRIBUTION ===")
    for note, count in pattern_counts.items():
        print(f"  {note}: {count} ({count/len(df)*100:.2f}%)")
    print()

    unique_patterns = [p for p in df["label_note"].unique() if p != "normal"]
    if len(unique_patterns) > 1:
        for pattern in sorted(unique_patterns):
            pat_rows = df[df["label_note"] == pattern]
            print(f"=== FEATURE STATS: {pattern.upper()} POSITIVES ===")
            print(pat_rows[feat_cols].describe().round(6).to_string())
            print()

# Check if peel wallets are completely invisible to graph features
train_pos = df[(df["split"] == "train") & (df["aml_label"] == 1)]
train_neg = df[(df["split"] == "train") & (df["aml_label"] == 0)]

print("=== ZERO-DEGREE ANALYSIS (train positives) ===")
for col in ["from_in_degree", "from_out_degree", "from_pagerank", "from_betweenness",
            "to_in_degree", "to_out_degree", "to_pagerank", "to_betweenness"]:
    zeros = (train_pos[col] == 0).sum()
    total = len(train_pos)
    print(f"  {col}==0 : {zeros}/{total} ({zeros/total*100:.1f}%)")

print()
print("=== ZERO-DEGREE ANALYSIS (train negatives) ===")
for col in ["from_in_degree", "from_out_degree", "from_pagerank", "from_betweenness",
            "to_in_degree", "to_out_degree", "to_pagerank", "to_betweenness"]:
    zeros = (train_neg[col] == 0).sum()
    total = len(train_neg)
    print(f"  {col}==0 : {zeros}/{total} ({zeros/total*100:.1f}%)")

print()
print("=== AMOUNT DISTRIBUTION ===")
print(f"Positives: min={positives['amount'].min():.4f}, max={positives['amount'].max():.4f}, mean={positives['amount'].mean():.4f}")
print(f"Negatives: min={negatives['amount'].min():.4f}, max={negatives['amount'].max():.4f}, mean={negatives['amount'].mean():.4f}")

# Check if positive wallets have non-zero degrees at all
print()
print("=== POSITIVES WITH ANY NON-ZERO GRAPH FEATURES ===")
any_nonzero = (
    (positives["from_in_degree"] > 0) |
    (positives["from_out_degree"] > 0) |
    (positives["from_pagerank"] > 0) |
    (positives["to_in_degree"] > 0) |
    (positives["to_out_degree"] > 0) |
    (positives["to_pagerank"] > 0)
)
print(f"  Positives with at least one non-zero graph feature: {any_nonzero.sum()}/{len(positives)} ({any_nonzero.sum()/len(positives)*100:.1f}%)")

# Per-pattern non-zero analysis for multi-pattern
if "label_note" in df.columns:
    unique_patterns = [p for p in df["label_note"].unique() if p != "normal"]
    if len(unique_patterns) > 1:
        for pattern in sorted(unique_patterns):
            pat_pos = df[df["label_note"] == pattern]
            pat_nonzero = (
                (pat_pos["from_in_degree"] > 0) |
                (pat_pos["from_out_degree"] > 0) |
                (pat_pos["from_pagerank"] > 0) |
                (pat_pos["to_in_degree"] > 0) |
                (pat_pos["to_out_degree"] > 0) |
                (pat_pos["to_pagerank"] > 0)
            )
            print(f"  {pattern}: {pat_nonzero.sum()}/{len(pat_pos)} ({pat_nonzero.sum()/len(pat_pos)*100:.1f}%)")

# Block time range info
print()
print("=== BLOCK_TIME RANGE ===")
df["block_time_dt"] = pd.to_datetime(df["block_time"], utc=True, errors="coerce")
print(f"Min: {df['block_time_dt'].min()}")
print(f"Max: {df['block_time_dt'].max()}")
print(f"Unique dates: {df['block_time_dt'].dt.date.nunique()}")
print(f"Unique from-wallets: {df['from'].nunique()}")
print(f"Unique to-wallets: {df['to'].nunique()}")
print(f"Unique tx_hashes: {df['tx_hash'].nunique()}")

# Split summary
print()
print("=== SPLIT SUMMARY ===")
for split_name in ["train", "val", "test"]:
    split_rows = df[df["split"] == split_name]
    pos = (split_rows["aml_label"] == 1).sum()
    neg = (split_rows["aml_label"] == 0).sum()
    total = len(split_rows)
    rate = pos / total if total > 0 else 0
    print(f"  {split_name}: {total} rows | pos={pos} neg={neg} | rate={rate:.4f}")

    # Per-pattern in split
    if "label_note" in df.columns:
        unique_patterns = [p for p in split_rows["label_note"].unique() if p != "normal"]
        if len(unique_patterns) > 1:
            for pattern in sorted(unique_patterns):
                pcount = (split_rows["label_note"] == pattern).sum()
                print(f"    {pattern}: {pcount}")
