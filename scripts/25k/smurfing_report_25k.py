"""Smurfing cluster report for the 25k multi-pattern dataset.

Extracts smurfing transactions from the 25k augmented dataset,
reconstructs fan-out/fan-in cluster topology, prints a detailed
terminal report, and saves visualisation PNGs to
``reports/25k/Smurfing Visualization/``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# Data loading
# ---------------------------------------------------------------------------

def _load_smurf_rows(
    transactions_path: Path,
    labels_path: Path | None,
) -> pd.DataFrame:
    df = pd.read_csv(transactions_path)

    # Multi-pattern augmented file has a ``pattern`` column.
    if "pattern" in df.columns:
        return df[df["pattern"].astype(str).str.lower() == "smurfing"].copy()

    # Fall back to the label file.
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(
            "No smurfing indicator found in transactions file and no label file provided."
        )

    labels = pd.read_csv(labels_path)
    if "label_note" not in labels.columns:
        raise ValueError("Label file must have a label_note column to identify smurfing rows.")

    smurf_hashes = set(
        labels.loc[labels["label_note"].astype(str).str.lower() == "smurfing", "tx_hash"].astype(str)
    )
    return df[df["tx_hash"].astype(str).isin(smurf_hashes)].copy()


# ---------------------------------------------------------------------------
# Cluster reconstruction
# ---------------------------------------------------------------------------

def _build_clusters(smurf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct smurfing clusters from the flat transaction table.

    Smurfing topology:  Source -> Mule_1 -> Collector
                        Source -> Mule_2 -> Collector
                        ...

    A source wallet is one that only appears in the ``from`` column within the
    smurfing subset and sends to multiple distinct intermediaries.  A collector
    is the shared destination of the mules.
    """
    if smurf_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    smurf_df = smurf_df.copy()
    smurf_df["amount"] = pd.to_numeric(smurf_df["amount"], errors="coerce")

    # Identify source wallets: wallets that appear as ``from`` but NOT as ``to``
    # within the smurfing subset, and that fan out to multiple destinations.
    all_from = set(smurf_df["from"].astype(str))
    all_to = set(smurf_df["to"].astype(str))

    # Sources: wallets that only send (not receive) within the smurfing subset.
    candidate_sources = all_from - all_to

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    cluster_id = 0

    for source in sorted(candidate_sources):
        fanout = smurf_df[smurf_df["from"].astype(str) == source]
        mules = set(fanout["to"].astype(str))

        if len(mules) < 2:
            # Not a cluster — need at least 2 mule wallets.
            continue

        cluster_id += 1

        # Fan-in: mules -> collector(s)
        fanin = smurf_df[smurf_df["from"].astype(str).isin(mules)]
        collectors = set(fanin["to"].astype(str))

        total_fanout_amount = float(fanout["amount"].sum())
        total_fanin_amount = float(fanin["amount"].sum())
        loss_pct = (
            ((total_fanout_amount - total_fanin_amount) / total_fanout_amount * 100)
            if total_fanout_amount > 0
            else 0.0
        )

        # Detail rows for fan-out leg.
        for _, row in fanout.iterrows():
            detail_rows.append({
                "cluster_id": cluster_id,
                "leg": "fan_out",
                "from": str(row["from"]),
                "to": str(row["to"]),
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else float("nan"),
                "tx_hash": row.get("tx_hash", ""),
                "block_number": row.get("block_number", ""),
                "block_time": row.get("block_time", ""),
            })

        # Detail rows for fan-in leg.
        for _, row in fanin.iterrows():
            detail_rows.append({
                "cluster_id": cluster_id,
                "leg": "fan_in",
                "from": str(row["from"]),
                "to": str(row["to"]),
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else float("nan"),
                "tx_hash": row.get("tx_hash", ""),
                "block_number": row.get("block_number", ""),
                "block_time": row.get("block_time", ""),
            })

        summary_rows.append({
            "cluster_id": cluster_id,
            "source_wallet": source,
            "num_mules": len(mules),
            "num_collectors": len(collectors),
            "collectors": ", ".join(sorted(collectors)[:3]),
            "total_fanout_amount": total_fanout_amount,
            "total_fanin_amount": total_fanin_amount,
            "loss_pct": loss_pct,
            "fanout_tx_count": len(fanout),
            "fanin_tx_count": len(fanin),
            "total_tx_count": len(fanout) + len(fanin),
        })

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    return details, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_cluster_size_distribution(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    counts = summary["num_mules"].value_counts().sort_index()
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(
        counts.index.astype(str),
        counts.values,
        color="#e76f51",
        alpha=0.9,
        label="Cluster count",
    )
    plt.title("25k Dataset — Smurfing Cluster Size Distribution (Mules per Cluster)")
    plt.xlabel("Number of Mule Wallets")
    plt.ylabel("Number of Clusters")
    plt.grid(axis="y", alpha=0.25)

    total = counts.values.sum()
    for bar, value in zip(bars, counts.values):
        pct = (value / total) * 100 if total else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{int(value)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_fanout_vs_fanin(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(
        summary["total_fanout_amount"],
        summary["total_fanin_amount"],
        c="#264653",
        alpha=0.6,
        s=summary["num_mules"] * 25,
        edgecolors="white",
        linewidths=0.5,
    )

    max_val = max(summary["total_fanout_amount"].max(), summary["total_fanin_amount"].max())
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.4, label="Break-even line")
    plt.title("25k Dataset — Smurfing Fan-out vs Fan-in Amount per Cluster")
    plt.xlabel("Total Fan-out Amount (Source -> Mules)")
    plt.ylabel("Total Fan-in Amount (Mules -> Collector)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_loss_distribution(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    plt.figure(figsize=(9, 5.5))
    plt.hist(
        summary["loss_pct"].dropna(),
        bins=25,
        color="#2a9d8f",
        alpha=0.85,
        edgecolor="white",
    )
    plt.title("25k Dataset — Forward Loss Distribution Across Smurfing Clusters")
    plt.xlabel("Loss % (fan-out - fan-in) / fan-out")
    plt.ylabel("Number of Clusters")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_cluster_topology(details: pd.DataFrame, summary: pd.DataFrame, out_path: Path, max_clusters: int = 10) -> None:
    """Horizontal dot-plot showing fan-out + fan-in legs per cluster."""
    if details.empty or summary.empty:
        return

    top = summary.sort_values("num_mules", ascending=False).head(max_clusters)
    keep = set(top["cluster_id"].tolist())
    plot_df = details[details["cluster_id"].isin(keep)].copy()

    plt.figure(figsize=(13, 7.5))
    colors = {"fan_out": "#e76f51", "fan_in": "#2a9d8f"}

    for cluster_id, grp in plot_df.groupby("cluster_id"):
        fanout_grp = grp[grp["leg"] == "fan_out"].reset_index()
        fanin_grp = grp[grp["leg"] == "fan_in"].reset_index()

        # Plot fan-out as left-side dots, fan-in as right-side dots.
        for i, (_, row) in enumerate(fanout_grp.iterrows()):
            plt.plot(i + 1, cluster_id, "o", color=colors["fan_out"], markersize=6, alpha=0.8)
        offset = len(fanout_grp)
        for i, (_, row) in enumerate(fanin_grp.iterrows()):
            plt.plot(offset + i + 1, cluster_id, "s", color=colors["fan_in"], markersize=6, alpha=0.8)

    plt.title("25k Dataset — Smurfing Cluster Topology (Sampled Clusters)")
    plt.xlabel("Transaction Index within Cluster")
    plt.ylabel("Cluster ID")
    plt.grid(alpha=0.25)

    legend_handles = [
        mpatches.Patch(color=colors["fan_out"], label="Fan-out (Source->Mule)"),
        mpatches.Patch(color=colors["fan_in"], label="Fan-in (Mule->Collector)"),
    ]
    plt.legend(handles=legend_handles, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def _short_addr(addr: str, head: int = 8, tail: int = 6) -> str:
    addr = str(addr)
    if len(addr) <= head + tail + 3:
        return addr
    return f"{addr[:head]}...{addr[-tail:]}"


def _print_terminal_report(details: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 80)
    print("SMURFING CLUSTER REPORT — 25K MULTI-PATTERN DATASET")
    print("=" * 80)
    print(f"Total clusters: {len(summary)}")
    print(f"Total smurfing transactions: {len(details)}")

    if summary.empty:
        print("No smurfing clusters found.")
        return

    print(f"Mules per cluster range: {int(summary['num_mules'].min())} to {int(summary['num_mules'].max())}")
    print(f"Transactions per cluster range: {int(summary['total_tx_count'].min())} to {int(summary['total_tx_count'].max())}")
    print(f"Mean forward loss: {summary['loss_pct'].mean():.2f}%")

    mule_counts = summary["num_mules"].value_counts().sort_index()
    print("Mule-count distribution (num_mules -> clusters):")
    for mule_count, count in mule_counts.items():
        print(f"  {int(mule_count)} -> {int(count)}")

    print("-" * 80)
    print("CLUSTER SUMMARY TABLE")
    print("-" * 80)
    header = (
        f"{'ID':>3} | {'Mules':>5} | {'Coll':>4} | {'Fan-out TX':>10} | {'Fan-in TX':>9} | "
        f"{'Fan-out Amt':>12} | {'Fan-in Amt':>12} | {'Loss%':>6}"
    )
    print(header)
    print("-" * len(header))

    for _, row in summary.sort_values("cluster_id").iterrows():
        print(
            f"{int(row['cluster_id']):>3} | {int(row['num_mules']):>5} | {int(row['num_collectors']):>4} | "
            f"{int(row['fanout_tx_count']):>10} | {int(row['fanin_tx_count']):>9} | "
            f"{float(row['total_fanout_amount']):>12.6f} | {float(row['total_fanin_amount']):>12.6f} | "
            f"{float(row['loss_pct']):>6.2f}"
        )

    print("-" * 80)
    print("CLUSTER DETAILS (first 20 clusters)")
    print("-" * 80)

    for _, row in summary.sort_values("cluster_id").head(20).iterrows():
        cid = int(row["cluster_id"])
        cluster_txs = details[details["cluster_id"] == cid]
        fanout = cluster_txs[cluster_txs["leg"] == "fan_out"]
        fanin = cluster_txs[cluster_txs["leg"] == "fan_in"]

        print(
            f"Cluster {cid} | source={_short_addr(row['source_wallet'])} | "
            f"mules={int(row['num_mules'])} | collectors={int(row['num_collectors'])} | "
            f"loss={float(row['loss_pct']):.2f}%"
        )
        print("  Fan-out (Source -> Mules):")
        for _, tx in fanout.iterrows():
            print(
                f"    {_short_addr(tx['from'])} -> {_short_addr(tx['to'])} | "
                f"amount={float(tx['amount']):.8f}"
            )
        print("  Fan-in (Mules -> Collector):")
        for _, tx in fanin.iterrows():
            print(
                f"    {_short_addr(tx['from'])} -> {_short_addr(tx['to'])} | "
                f"amount={float(tx['amount']):.8f}"
            )
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smurfing cluster report for the 25k multi-pattern dataset.")
    parser.add_argument(
        "--transactions",
        default="data/processed/25k/augmented_transactions_multipattern.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/25k/transaction_labels_multipattern.csv",
    )
    parser.add_argument("--out-dir", default="reports/25k/Smurfing Visualization")
    args = parser.parse_args()

    out_dir = _resolve_user_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions_path = _resolve_user_path(args.transactions)
    labels_path = _resolve_user_path(args.labels) if args.labels else None

    smurf_df = _load_smurf_rows(transactions_path, labels_path)
    details, summary = _build_clusters(smurf_df)

    _print_terminal_report(details, summary)

    size_dist_plot = out_dir / "smurfing_cluster_size_distribution_25k.png"
    fanout_vs_fanin_plot = out_dir / "smurfing_fanout_vs_fanin_25k.png"
    loss_dist_plot = out_dir / "smurfing_loss_distribution_25k.png"
    topology_plot = out_dir / "smurfing_cluster_topology_25k.png"

    _plot_cluster_size_distribution(summary, size_dist_plot)
    _plot_fanout_vs_fanin(summary, fanout_vs_fanin_plot)
    _plot_loss_distribution(summary, loss_dist_plot)
    _plot_cluster_topology(details, summary, topology_plot)

    print("=" * 80)
    print("IMAGES SAVED")
    print("=" * 80)
    print(size_dist_plot)
    print(fanout_vs_fanin_plot)
    print(loss_dist_plot)
    print(topology_plot)


if __name__ == "__main__":
    main()
