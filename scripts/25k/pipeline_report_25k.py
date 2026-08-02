"""Pipeline overview report for the 25k multi-pattern dataset.

Generates a comprehensive terminal report and visualisation PNGs covering:
  - Dataset composition (base vs peel-chain vs smurfing)
  - Label distribution across train/val/test splits
  - Amount and graph feature statistics per class
  - Wallet overlap between injected patterns and real transactions
  - Feature correlation heatmap

Saves images to ``reports/25k/Pipeline Overview/``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def _resolve_user_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# Dataset composition
# ---------------------------------------------------------------------------

def _composition_report(aug_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary of the dataset composition by pattern type."""
    rows = []

    total = len(aug_df)

    if "label_note" in labels_df.columns:
        note_counts = labels_df["label_note"].value_counts()
        for note, count in note_counts.items():
            rows.append({
                "pattern": str(note),
                "tx_count": int(count),
                "pct_of_total": round(count / total * 100, 2) if total else 0,
            })
    else:
        # Fallback: just positive vs negative.
        pos = int(labels_df["aml_label"].astype(int).sum())
        neg = int(len(labels_df) - pos)
        rows.append({"pattern": "normal", "tx_count": neg, "pct_of_total": round(neg / total * 100, 2)})
        rows.append({"pattern": "suspicious", "tx_count": pos, "pct_of_total": round(pos / total * 100, 2)})

    return pd.DataFrame(rows).sort_values("tx_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split distribution
# ---------------------------------------------------------------------------

def _split_report(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-split summary with label counts and positive rates."""
    rows = []
    for split_name in ["train", "val", "test"]:
        subset = modeling_df[modeling_df["split"] == split_name]
        n = len(subset)
        pos = int(subset["aml_label"].astype(int).sum())
        neg = n - pos
        pos_rate = round(pos / n * 100, 2) if n else 0
        rows.append({
            "split": split_name,
            "total": n,
            "label_0": neg,
            "label_1": pos,
            "positive_rate_%": pos_rate,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Amount statistics
# ---------------------------------------------------------------------------

def _amount_stats(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Per-class amount statistics."""
    rows = []
    for label in [0, 1]:
        subset = modeling_df[modeling_df["aml_label"] == label]["amount"]
        rows.append({
            "aml_label": label,
            "count": len(subset),
            "mean": round(float(subset.mean()), 8) if len(subset) else 0,
            "std": round(float(subset.std()), 8) if len(subset) > 1 else 0,
            "min": round(float(subset.min()), 8) if len(subset) else 0,
            "median": round(float(subset.median()), 8) if len(subset) else 0,
            "max": round(float(subset.max()), 8) if len(subset) else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graph feature statistics
# ---------------------------------------------------------------------------

def _graph_feature_stats(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Per-class mean of graph-derived features."""
    graph_cols = [
        c for c in modeling_df.columns
        if c.startswith(("from_", "to_")) and c.endswith(("degree", "flow_ratio", "pagerank", "betweenness"))
    ]
    if not graph_cols:
        return pd.DataFrame()

    rows = []
    for label in [0, 1]:
        subset = modeling_df[modeling_df["aml_label"] == label]
        stats = {"aml_label": label}
        for col in graph_cols:
            stats[col + "_mean"] = round(float(subset[col].mean()), 6) if len(subset) else 0
        rows.append(stats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_composition_pie(composition: pd.DataFrame, out_path: Path) -> None:
    if composition.empty:
        return

    labels = composition["pattern"].tolist()
    sizes = composition["tx_count"].tolist()
    colors = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261"][:len(labels)]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    plt.title("25k Dataset — Transaction Composition by Pattern", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_split_bars(split_df: pd.DataFrame, out_path: Path) -> None:
    if split_df.empty:
        return

    x = np.arange(len(split_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, split_df["label_0"], width, label="Normal (0)", color="#264653", alpha=0.9)
    ax.bar(x + width / 2, split_df["label_1"], width, label="Suspicious (1)", color="#e76f51", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(split_df["split"].tolist())
    ax.set_xlabel("Split")
    ax.set_ylabel("Transaction Count")
    ax.set_title("25k Dataset — Label Distribution Across Splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # Annotate bars.
    for i, row in split_df.iterrows():
        ax.text(i - width / 2, row["label_0"] + 20, str(row["label_0"]), ha="center", fontsize=8)
        ax.text(i + width / 2, row["label_1"] + 20, str(row["label_1"]), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_amount_distribution(modeling_df: pd.DataFrame, out_path: Path) -> None:
    if modeling_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    normal = modeling_df[modeling_df["aml_label"] == 0]["amount"].dropna()
    suspicious = modeling_df[modeling_df["aml_label"] == 1]["amount"].dropna()

    if not normal.empty:
        ax.hist(
            np.log1p(normal.clip(lower=0)),
            bins=60,
            alpha=0.7,
            color="#264653",
            label=f"Normal (n={len(normal)})",
            density=True,
        )
    if not suspicious.empty:
        ax.hist(
            np.log1p(suspicious.clip(lower=0)),
            bins=60,
            alpha=0.7,
            color="#e76f51",
            label=f"Suspicious (n={len(suspicious)})",
            density=True,
        )

    ax.set_title("25k Dataset — Amount Distribution (log1p) by Class")
    ax.set_xlabel("log1p(amount)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_feature_correlation(modeling_df: pd.DataFrame, out_path: Path) -> None:
    """Correlation heatmap of numeric modelling features."""
    drop_cols = {
        "transaction_id", "tx_hash", "split", "block_time", "from", "to", "label_note",
    }
    numeric_cols = [
        c for c in modeling_df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(modeling_df[c])
    ]
    if len(numeric_cols) < 2:
        return

    corr = modeling_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(numeric_cols, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("25k Dataset — Feature Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_positive_rate_by_pattern(labels_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart showing per-pattern transaction counts."""
    if "label_note" not in labels_df.columns:
        return

    counts = labels_df["label_note"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 5.5))
    colors = {"normal": "#264653", "peel_chain": "#2a9d8f", "smurfing": "#e76f51"}
    bar_colors = [colors.get(str(k), "#5e6472") for k in counts.index]
    bars = plt.bar(counts.index.astype(str), counts.values, color=bar_colors, alpha=0.9)
    plt.title("25k Dataset — Transaction Count by Pattern Label")
    plt.xlabel("Pattern")
    plt.ylabel("Number of Transactions")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def _print_terminal_report(
    aug_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    modeling_df: pd.DataFrame | None,
    composition: pd.DataFrame,
    split_report: pd.DataFrame | None,
    amount_stats_df: pd.DataFrame | None,
    graph_stats_df: pd.DataFrame | None,
    metadata: dict | None,
) -> None:
    print("=" * 88)
    print("25K MULTI-PATTERN PIPELINE OVERVIEW REPORT")
    print("=" * 88)

    # Dataset composition.
    print(f"\nTotal augmented transactions: {len(aug_df)}")
    print(f"Total label rows: {len(labels_df)}")
    print("\n--- COMPOSITION BY PATTERN ---")
    print(composition.to_string(index=False))

    # Metadata.
    if metadata:
        print("\n--- PREP METADATA ---")
        for key, val in metadata.items():
            if key != "splits":
                print(f"  {key}: {val}")

    # Split distribution.
    if split_report is not None and not split_report.empty:
        print("\n--- SPLIT DISTRIBUTION ---")
        print(split_report.to_string(index=False))

    # Amount statistics.
    if amount_stats_df is not None and not amount_stats_df.empty:
        print("\n--- AMOUNT STATISTICS BY CLASS ---")
        print(amount_stats_df.to_string(index=False))

    # Graph feature statistics.
    if graph_stats_df is not None and not graph_stats_df.empty:
        print("\n--- GRAPH FEATURE MEANS BY CLASS ---")
        print(graph_stats_df.to_string(index=False))

    # Pattern-specific summaries.
    if "label_note" in labels_df.columns:
        for pattern in ["peel_chain", "smurfing"]:
            pattern_rows = labels_df[labels_df["label_note"].astype(str) == pattern]
            if len(pattern_rows) > 0:
                hashes = set(pattern_rows["tx_hash"].astype(str))
                pattern_txs = aug_df[aug_df["tx_hash"].astype(str).isin(hashes)]
                unique_from = pattern_txs["from"].nunique() if "from" in pattern_txs.columns else 0
                unique_to = pattern_txs["to"].nunique() if "to" in pattern_txs.columns else 0
                unique_wallets = set(pattern_txs["from"].tolist()) | set(pattern_txs["to"].tolist())

                # Overlap with base (normal) wallets.
                normal_hashes = set(
                    labels_df.loc[labels_df["label_note"].astype(str) == "normal", "tx_hash"].astype(str)
                )
                normal_txs = aug_df[aug_df["tx_hash"].astype(str).isin(normal_hashes)]
                normal_wallets = set(normal_txs["from"].tolist()) | set(normal_txs["to"].tolist())
                overlap = unique_wallets & normal_wallets

                print(f"\n--- {pattern.upper()} INJECTION SUMMARY ---")
                print(f"  Transactions: {len(pattern_rows)}")
                print(f"  Unique senders: {unique_from}")
                print(f"  Unique receivers: {unique_to}")
                print(f"  Unique wallets: {len(unique_wallets)}")
                print(f"  Wallets overlapping with normal graph: {len(overlap)} ({len(overlap) / max(len(unique_wallets), 1) * 100:.1f}%)")

                if "amount" in pattern_txs.columns:
                    amounts = pd.to_numeric(pattern_txs["amount"], errors="coerce").dropna()
                    if not amounts.empty:
                        print(f"  Amount — mean: {amounts.mean():.8f}, median: {amounts.median():.8f}, "
                              f"min: {amounts.min():.8f}, max: {amounts.max():.8f}")

    print("\n" + "=" * 88)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline overview report for the 25k multi-pattern dataset.")
    parser.add_argument(
        "--transactions",
        default="data/processed/25k/augmented_transactions_multipattern.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/25k/transaction_labels_multipattern.csv",
    )
    parser.add_argument(
        "--modeling",
        default="data/processed/25k/modeling_dataset_transactions_25k_multi.csv",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/25k/model_prep_metadata_25k_multi.json",
    )
    parser.add_argument("--out-dir", default="reports/25k/Pipeline Overview")
    args = parser.parse_args()

    out_dir = _resolve_user_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data.
    aug_path = _resolve_user_path(args.transactions)
    labels_path = _resolve_user_path(args.labels)
    modeling_path = _resolve_user_path(args.modeling)
    metadata_path = _resolve_user_path(args.metadata)

    aug_df = pd.read_csv(aug_path)
    labels_df = pd.read_csv(labels_path)

    modeling_df = None
    if modeling_path.exists():
        modeling_df = pd.read_csv(modeling_path)

    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Build report tables.
    composition = _composition_report(aug_df, labels_df)
    split_report = _split_report(modeling_df) if modeling_df is not None else None
    amount_stats_df = _amount_stats(modeling_df) if modeling_df is not None else None
    graph_stats_df = _graph_feature_stats(modeling_df) if modeling_df is not None else None

    # Terminal output.
    _print_terminal_report(
        aug_df, labels_df, modeling_df,
        composition, split_report, amount_stats_df, graph_stats_df, metadata,
    )

    # Plots.
    _plot_composition_pie(composition, out_dir / "composition_pie_25k.png")
    _plot_positive_rate_by_pattern(labels_df, out_dir / "pattern_tx_counts_25k.png")

    if modeling_df is not None:
        _plot_split_bars(split_report, out_dir / "split_distribution_25k.png")
        _plot_amount_distribution(modeling_df, out_dir / "amount_distribution_25k.png")
        _plot_feature_correlation(modeling_df, out_dir / "feature_correlation_25k.png")

    print("IMAGES SAVED")
    print("=" * 88)
    for f in sorted(out_dir.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
