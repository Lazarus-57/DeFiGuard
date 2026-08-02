"""Comprehensive dataset integrity and validity verification."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR

parser = argparse.ArgumentParser(description="Verify dataset integrity.")
parser.add_argument("pipeline", choices=["5k", "25k", "25k-multi", "100k-multi"], help="Which dataset to verify")
args = parser.parse_args()

if args.pipeline == "5k":
    AUGMENTED = PROJECT_ROOT / "data" / "processed" / "5k" / "augmented_transactions_peel.csv"
    LABELS = PROJECT_ROOT / "data" / "processed" / "5k" / "transaction_labels_peel.csv"
    MODELING = PROJECT_ROOT / "data" / "processed" / "5k" / "modeling_dataset_transactions.csv"
    METADATA = PROJECT_ROOT / "data" / "processed" / "5k" / "model_prep_metadata.json"
    SPLIT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "5k" / "modeling_split_summary.csv"
    RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "5k" / "eth_transfers_5k_2024.csv"
elif args.pipeline == "25k":
    AUGMENTED = PROJECT_ROOT / "data" / "processed" / "25k" / "augmented_transactions_peel_25k.csv"
    LABELS = PROJECT_ROOT / "data" / "processed" / "25k" / "transaction_labels_peel_25k.csv"
    MODELING = PROJECT_ROOT / "data" / "processed" / "25k" / "modeling_dataset_transactions_25k.csv"
    METADATA = PROJECT_ROOT / "data" / "processed" / "25k" / "model_prep_metadata_25k.json"
    SPLIT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "25k" / "modeling_split_summary_25k.csv"
    RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "25k" / "eth_transfers_25k_2024_final.csv"
elif args.pipeline == "25k-multi":
    AUGMENTED = PROJECT_ROOT / "data" / "processed" / "25k" / "augmented_transactions_multipattern.csv"
    LABELS = PROJECT_ROOT / "data" / "processed" / "25k" / "transaction_labels_multipattern.csv"
    MODELING = PROJECT_ROOT / "data" / "processed" / "25k" / "modeling_dataset_transactions_25k_multi.csv"
    METADATA = PROJECT_ROOT / "data" / "processed" / "25k" / "model_prep_metadata_25k_multi.json"
    SPLIT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "25k" / "modeling_split_summary_25k_multi.csv"
    RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "25k" / "eth_transfers_25k_2024_final.csv"
else:
    AUGMENTED = PROJECT_ROOT / "data" / "processed" / "100k" / "augmented_transactions_multipattern.csv"
    LABELS = PROJECT_ROOT / "data" / "processed" / "100k" / "transaction_labels_multipattern.csv"
    MODELING = PROJECT_ROOT / "data" / "processed" / "100k" / "modeling_dataset_transactions_100k_multi.csv"
    METADATA = PROJECT_ROOT / "data" / "processed" / "100k" / "model_prep_metadata_100k_multi.json"
    SPLIT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "100k" / "modeling_split_summary_100k_multi.csv"
    RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "100k" / "eth_transfers_100k_2024.csv"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = []
warnings = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"{name}: {detail}" if detail else name
        failures.append(msg)
        print(f"  {FAIL} {name} — {detail}")


def warn(name: str, detail: str) -> None:
    warnings.append(f"{name}: {detail}")
    print(f"  {WARN} {name} — {detail}")


print("=" * 80)
print("DATASET INTEGRITY & VALIDITY VERIFICATION")
print("25k Multipattern (Peel Chain + Smurfing)")
print("=" * 80)

# ============================================================
# 1. FILE EXISTENCE
# ============================================================
print("\n--- 1. FILE EXISTENCE ---")
for label, path in [
    ("Raw source", RAW_SOURCE),
    ("Augmented transactions", AUGMENTED),
    ("Labels", LABELS),
    ("Modeling dataset", MODELING),
    ("Metadata", METADATA),
    ("Split summary", SPLIT_SUMMARY),
]:
    check(f"{label} exists", path.exists(), f"Missing: {path}")

if not all(p.exists() for p in [AUGMENTED, LABELS, MODELING, METADATA]):
    print("\nCritical files missing. Cannot continue.")
    sys.exit(1)

# ============================================================
# 2. LOAD ALL FILES
# ============================================================
print("\n--- 2. LOAD & BASIC SHAPE ---")
aug_df = pd.read_csv(AUGMENTED)
labels_df = pd.read_csv(LABELS)
model_df = pd.read_csv(MODELING)
meta = json.loads(METADATA.read_text(encoding="utf-8"))
raw_df = pd.read_csv(RAW_SOURCE)

check("Augmented CSV loads", len(aug_df) > 0, f"rows={len(aug_df)}")
check("Labels CSV loads", len(labels_df) > 0, f"rows={len(labels_df)}")
check("Modeling CSV loads", len(model_df) > 0, f"rows={len(model_df)}")
print(f"       Augmented rows: {len(aug_df)}")
print(f"       Labels rows: {len(labels_df)}")
print(f"       Modeling rows: {len(model_df)}")
print(f"       Raw source rows: {len(raw_df)}")

# ============================================================
# 3. ROW COUNT CONSISTENCY
# ============================================================
print("\n--- 3. ROW COUNT CONSISTENCY ---")
check("Augmented == Labels row count", len(aug_df) == len(labels_df),
      f"aug={len(aug_df)} vs labels={len(labels_df)}")
check("Augmented == Modeling row count", len(aug_df) == len(model_df),
      f"aug={len(aug_df)} vs model={len(model_df)}")
check("Metadata row_count matches", meta["row_count"] == len(model_df),
      f"meta={meta['row_count']} vs actual={len(model_df)}")

base_count = len(raw_df)
peel_count = (labels_df["label_note"] == "peel_chain").sum()
smurf_count = (labels_df["label_note"] == "smurfing").sum()
normal_count = (labels_df["label_note"] == "normal").sum()
check("Base + peel + smurf == total", base_count + peel_count + smurf_count == len(aug_df),
      f"{base_count} + {peel_count} + {smurf_count} = {base_count + peel_count + smurf_count} vs {len(aug_df)}")
check("Normal labels == base count", normal_count == base_count,
      f"normal={normal_count} vs raw={base_count}")

# ============================================================
# 4. TX_HASH INTEGRITY
# ============================================================
print("\n--- 4. TX_HASH INTEGRITY ---")
check("No null tx_hash in augmented", aug_df["tx_hash"].isna().sum() == 0,
      f"nulls={aug_df['tx_hash'].isna().sum()}")
check("No null tx_hash in labels", labels_df["tx_hash"].isna().sum() == 0,
      f"nulls={labels_df['tx_hash'].isna().sum()}")
check("No null tx_hash in modeling", model_df["tx_hash"].isna().sum() == 0,
      f"nulls={model_df['tx_hash'].isna().sum()}")

aug_dups = aug_df["tx_hash"].duplicated().sum()
label_dups = labels_df["tx_hash"].duplicated().sum()
model_dups = model_df["tx_hash"].duplicated().sum()
check("No duplicate tx_hash in augmented", aug_dups == 0, f"dups={aug_dups}")
check("No duplicate tx_hash in labels", label_dups == 0, f"dups={label_dups}")
check("No duplicate tx_hash in modeling", model_dups == 0, f"dups={model_dups}")

# Cross-check: every augmented tx_hash has a label
aug_hashes = set(aug_df["tx_hash"].astype(str))
label_hashes = set(labels_df["tx_hash"].astype(str))
model_hashes = set(model_df["tx_hash"].astype(str))
check("All augmented hashes have labels", aug_hashes == label_hashes,
      f"aug_only={len(aug_hashes - label_hashes)}, label_only={len(label_hashes - aug_hashes)}")
check("All modeling hashes in augmented", model_hashes.issubset(aug_hashes),
      f"model_only={len(model_hashes - aug_hashes)}")

# ============================================================
# 5. LABEL INTEGRITY
# ============================================================
print("\n--- 5. LABEL INTEGRITY ---")
check("aml_label has no nulls", model_df["aml_label"].isna().sum() == 0)
check("aml_label is binary (0/1 only)", set(model_df["aml_label"].unique()) == {0, 1},
      f"unique={sorted(model_df['aml_label'].unique())}")

model_pos = (model_df["aml_label"] == 1).sum()
model_neg = (model_df["aml_label"] == 0).sum()
check("Metadata label_0 matches", meta["label_0_count"] == model_neg,
      f"meta={meta['label_0_count']} vs actual={model_neg}")
check("Metadata label_1 matches", meta["label_1_count"] == model_pos,
      f"meta={meta['label_1_count']} vs actual={model_pos}")

# label_note consistency
label_notes = model_df["label_note"].unique()
check("label_note contains 'normal'", "normal" in label_notes, f"found: {label_notes}")
check("label_note contains 'peel_chain'", "peel_chain" in label_notes, f"found: {label_notes}")
if args.pipeline == "25k-multi":
    check("label_note contains 'smurfing'", "smurfing" in label_notes, f"found: {label_notes}")

peel_model = model_df[model_df["label_note"] == "peel_chain"]
smurf_model = model_df[model_df["label_note"] == "smurfing"]
normal_model = model_df[model_df["label_note"] == "normal"]
check("All peel_chain rows have aml_label=1", (peel_model["aml_label"] == 1).all())
check("All smurfing rows have aml_label=1", (smurf_model["aml_label"] == 1).all())
check("All normal rows have aml_label=0", (normal_model["aml_label"] == 0).all())
check("peel_chain count consistent", len(peel_model) == peel_count,
      f"model={len(peel_model)} vs labels={peel_count}")
check("smurfing count consistent", len(smurf_model) == smurf_count,
      f"model={len(smurf_model)} vs labels={smurf_count}")

# ============================================================
# 6. SPLIT INTEGRITY
# ============================================================
print("\n--- 6. SPLIT INTEGRITY ---")
check("split column exists", "split" in model_df.columns)
split_vals = set(model_df["split"].unique())
check("Exactly 3 splits (train/val/test)", split_vals == {"train", "val", "test"},
      f"found: {split_vals}")

train_df = model_df[model_df["split"] == "train"]
val_df = model_df[model_df["split"] == "val"]
test_df = model_df[model_df["split"] == "test"]

train_frac = len(train_df) / len(model_df)
val_frac = len(val_df) / len(model_df)
test_frac = len(test_df) / len(model_df)
check("Train ~70%", 0.68 <= train_frac <= 0.72, f"actual={train_frac:.4f}")
check("Val ~15%", 0.13 <= val_frac <= 0.17, f"actual={val_frac:.4f}")
check("Test ~15%", 0.13 <= test_frac <= 0.17, f"actual={test_frac:.4f}")

# Positives in every split
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    pos = (sdf["aml_label"] == 1).sum()
    check(f"{name} has positives", pos > 0, f"positives={pos}")

# Both patterns present in every split
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    has_peel = (sdf["label_note"] == "peel_chain").sum() > 0
    has_smurf = (sdf["label_note"] == "smurfing").sum() > 0
    check(f"{name} has peel_chain", has_peel)
    if args.pipeline == "25k-multi":
        check(f"{name} has smurfing", has_smurf)

# Temporal ordering: train < val < test
model_df_copy = model_df.copy()
model_df_copy["block_time_dt"] = pd.to_datetime(model_df_copy["block_time"], utc=True, errors="coerce")
train_max = model_df_copy[model_df_copy["split"] == "train"]["block_time_dt"].max()
val_min = model_df_copy[model_df_copy["split"] == "val"]["block_time_dt"].min()
val_max = model_df_copy[model_df_copy["split"] == "val"]["block_time_dt"].max()
test_min = model_df_copy[model_df_copy["split"] == "test"]["block_time_dt"].min()
check("Temporal: train_max <= val_min", train_max <= val_min,
      f"train_max={train_max}, val_min={val_min}")
check("Temporal: val_max <= test_min", val_max <= test_min,
      f"val_max={val_max}, test_min={test_min}")

# ============================================================
# 7. FEATURE INTEGRITY
# ============================================================
print("\n--- 7. FEATURE INTEGRITY ---")
required_features = [
    "amount", "amount_log1p", "block_number", "hour_of_day", "day_of_week",
    "from_in_degree", "from_out_degree", "from_flow_ratio", "from_pagerank", "from_betweenness",
    "to_in_degree", "to_out_degree", "to_flow_ratio", "to_pagerank", "to_betweenness",
]
for feat in required_features:
    check(f"Feature '{feat}' exists", feat in model_df.columns)

numeric_features = [f for f in required_features if f in model_df.columns]
for feat in numeric_features:
    col = pd.to_numeric(model_df[feat], errors="coerce")
    nulls = col.isna().sum()
    infs = np.isinf(col.fillna(0)).sum()
    check(f"No nulls/infs in '{feat}'", nulls == 0 and infs == 0,
          f"nulls={nulls}, infs={infs}")

# Amount sanity
check("No negative amounts", (model_df["amount"] >= 0).all(),
      f"negatives={(model_df['amount'] < 0).sum()}")

# Degree columns are non-negative integers
for col in ["from_in_degree", "from_out_degree", "to_in_degree", "to_out_degree"]:
    vals = model_df[col]
    check(f"'{col}' non-negative", (vals >= 0).all())

# ============================================================
# 8. LEAKAGE SAFETY
# ============================================================
print("\n--- 8. LEAKAGE SAFETY ---")

# Wallet features should come from train-only graph
# Check: wallets that appear ONLY in val/test should have zero graph features
val_test_only_from = set(model_df[model_df["split"].isin(["val", "test"])]["from"]) - set(train_df["from"]) - set(train_df["to"])
val_test_only_to = set(model_df[model_df["split"].isin(["val", "test"])]["to"]) - set(train_df["from"]) - set(train_df["to"])

if val_test_only_from:
    vt_from_rows = model_df[(model_df["from"].isin(val_test_only_from)) & (model_df["split"].isin(["val", "test"]))]
    from_feats_zero = (vt_from_rows[["from_in_degree", "from_out_degree", "from_pagerank", "from_betweenness"]] == 0).all(axis=1).all()
    check("Val/test-only 'from' wallets have zero graph features", from_feats_zero,
          f"Non-zero found for {len(vt_from_rows)} rows")
else:
    print(f"  {PASS} No val/test-only 'from' wallets (all appear in train)")

if val_test_only_to:
    vt_to_rows = model_df[(model_df["to"].isin(val_test_only_to)) & (model_df["split"].isin(["val", "test"]))]
    to_feats_zero = (vt_to_rows[["to_in_degree", "to_out_degree", "to_pagerank", "to_betweenness"]] == 0).all(axis=1).all()
    check("Val/test-only 'to' wallets have zero graph features", to_feats_zero,
          f"Non-zero found for {len(vt_to_rows)} rows")
else:
    print(f"  {PASS} No val/test-only 'to' wallets (all appear in train)")

# ============================================================
# 9. PATTERN-SPECIFIC VALIDATION
# ============================================================
print("\n--- 9. PATTERN-SPECIFIC VALIDATION ---")

# Peel chains: from_out_degree should be ~1 (linear chain)
if len(peel_model) > 0:
    peel_from_out_median = peel_model["from_out_degree"].median()
    check("Peel chain from_out_degree median ~1", 0 <= peel_from_out_median <= 2,
          f"median={peel_from_out_median}")

# Smurfing specific checks
if len(smurf_model) > 0:
    smurf_from_out_median = smurf_model["from_out_degree"].median()
    check("Smurfing from_out_degree > peel (fan-out)", smurf_from_out_median >= peel_from_out_median,
          f"smurf_median={smurf_from_out_median} vs peel_median={peel_from_out_median}")

    # Smurfing: to_in_degree should be higher (fan-in at collector)
    smurf_to_in_median = smurf_model["to_in_degree"].median()
    peel_to_in_median = peel_model["to_in_degree"].median()
    check("Smurfing to_in_degree >= peel (fan-in)", smurf_to_in_median >= peel_to_in_median,
          f"smurf_median={smurf_to_in_median} vs peel_median={peel_to_in_median}")
else:
    print(f"  {PASS} No smurfing rows found (skipping smurfing graph validation)")

# Positive rate in reasonable range
pos_rate = model_pos / len(model_df)
if args.pipeline == "25k-multi":
    check("Positive rate 5-10%", 0.05 <= pos_rate <= 0.10, f"actual={pos_rate:.4f}")
else:
    check("Positive rate 3-5%", 0.03 <= pos_rate <= 0.05, f"actual={pos_rate:.4f}")

# ============================================================
# 10. METADATA CONSISTENCY
# ============================================================
print("\n--- 10. METADATA CONSISTENCY ---")
check("No duplicate tx_hashes (meta)", meta["duplicate_tx_hashes_in_transactions"] == 0)
check("No duplicate label hashes (meta)", meta["duplicate_tx_hashes_in_labels"] == 0)
check("No orphan label hashes (meta)", meta["orphan_label_hashes"] == 0)

# Verify split summary file
split_summary = pd.read_csv(SPLIT_SUMMARY)
for _, row in split_summary.iterrows():
    split_name = row["split"]
    actual_total = len(model_df[model_df["split"] == split_name])
    check(f"Split summary '{split_name}' total matches", int(row["total_count"]) == actual_total,
          f"summary={int(row['total_count'])} vs actual={actual_total}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
if not failures:
    print(f"{PASS} ALL CHECKS PASSED — Dataset is ready for model training.")
else:
    print(f"{FAIL} {len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"    - {f}")

if warnings:
    print(f"\n{WARN} {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"    - {w}")

print("=" * 80)
