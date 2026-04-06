import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR


def _resolve_user_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_peel_rows(transactions_path: Path, labels_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(transactions_path)

    if "pattern" in df.columns:
        return df[df["pattern"].astype(str).str.lower() == "peel_chain"].copy()

    if "aml_label" in df.columns:
        return df[df["aml_label"].astype(int) == 1].copy()

    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(
            "No peel indicator found in transactions file and no label file provided."
        )

    labels = pd.read_csv(labels_path)
    peel_hashes = set(labels.loc[labels["aml_label"].astype(int) == 1, "tx_hash"].astype(str))
    return df[df["tx_hash"].astype(str).isin(peel_hashes)].copy()


def _build_chains(peel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if peel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    peel_df = peel_df.copy()
    peel_df["amount"] = pd.to_numeric(peel_df["amount"], errors="coerce")
    peel_df["block_number"] = pd.to_numeric(peel_df.get("block_number"), errors="coerce")
    peel_df = peel_df.sort_values(["block_number", "tx_hash"], ascending=[True, True]).reset_index(drop=True)

    outgoing: dict[str, list[int]] = {}
    incoming_count: dict[str, int] = {}

    for idx, row in peel_df.iterrows():
        src = str(row["from"])
        dst = str(row["to"])
        outgoing.setdefault(src, []).append(idx)
        incoming_count[dst] = incoming_count.get(dst, 0) + 1

    all_from = set(peel_df["from"].astype(str))
    all_to = set(peel_df["to"].astype(str))
    starts = sorted(all_from - all_to)

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for chain_id, start in enumerate(starts, start=1):
        current = start
        hop = 1
        visited = set()
        amounts = []
        first_amount = None
        last_amount = None
        min_drop_pct = None
        max_drop_pct = None
        anomaly = ""

        while True:
            if current in visited:
                anomaly = "cycle_detected"
                break
            visited.add(current)

            out_idxs = outgoing.get(current, [])
            if len(out_idxs) == 0:
                break
            if len(out_idxs) > 1:
                anomaly = "branching_detected"
                break

            row = peel_df.iloc[out_idxs[0]]
            next_wallet = str(row["to"])
            amount = float(row["amount"]) if pd.notna(row["amount"]) else float("nan")

            if first_amount is None:
                first_amount = amount
            last_amount = amount
            amounts.append(amount)

            drop_pct = None
            if len(amounts) > 1 and amounts[-2] not in (0, float("nan")):
                prev = amounts[-2]
                if prev != 0:
                    drop_pct = ((prev - amount) / prev) * 100
                    if min_drop_pct is None or drop_pct < min_drop_pct:
                        min_drop_pct = drop_pct
                    if max_drop_pct is None or drop_pct > max_drop_pct:
                        max_drop_pct = drop_pct

            detail_rows.append(
                {
                    "chain_id": chain_id,
                    "hop": hop,
                    "from": current,
                    "to": next_wallet,
                    "amount": amount,
                    "drop_pct_from_prev": drop_pct,
                    "tx_hash": row.get("tx_hash", ""),
                    "block_number": row.get("block_number", ""),
                    "block_time": row.get("block_time", ""),
                }
            )

            current = next_wallet
            hop += 1

        wallet_count = hop
        tx_count = hop - 1
        summary_rows.append(
            {
                "chain_id": chain_id,
                "start_wallet": start,
                "end_wallet": current,
                "wallet_count": wallet_count,
                "tx_count": tx_count,
                "start_amount": first_amount,
                "end_amount": last_amount,
                "min_drop_pct": min_drop_pct,
                "max_drop_pct": max_drop_pct,
                "anomaly": anomaly,
            }
        )

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    return details, summary


def _plot_chain_amount_profiles(details: pd.DataFrame, out_path: Path, max_chains: int = 15) -> None:
    if details.empty:
        return

    top = (
        details.groupby("chain_id", as_index=False)["hop"].max().sort_values("hop", ascending=False).head(max_chains)
    )
    keep = set(top["chain_id"].tolist())
    plot_df = details[details["chain_id"].isin(keep)].copy()

    plt.figure(figsize=(13, 7.5))
    for chain_id, grp in plot_df.groupby("chain_id"):
        grp = grp.sort_values("hop")
        plt.plot(
            grp["hop"],
            grp["amount"],
            marker="o",
            linewidth=1.8,
            alpha=0.9,
            label=f"Chain {chain_id}"
        )

    max_hop = int(plot_df["hop"].max()) if not plot_df.empty else 1
    plt.xticks(range(1, max_hop + 1))
    plt.title("Peel-Chain Amount Decay by Hop (Sampled Chains)")
    plt.xlabel("Hop Number (Transaction Position in Chain)")
    plt.ylabel("Transfer Amount")
    plt.grid(alpha=0.3)
    plt.legend(title="Chain ID", ncol=3, fontsize=8, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.figtext(
        0.01,
        0.01,
        "X-axis: hop number. Y-axis: transfer amount. Downward trend indicates peel-like decay.",
        fontsize=9,
        alpha=0.8,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_linear_chain_layout(details: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    if details.empty:
        return

    plt.figure(figsize=(14, 9))
    length_color = {4: "#2a9d8f", 5: "#e9c46a", 6: "#e76f51"}
    length_map = summary.set_index("chain_id")["wallet_count"].to_dict()

    for chain_id, grp in details.groupby("chain_id"):
        grp = grp.sort_values("hop")
        y = [chain_id] * len(grp)
        wallet_count = int(length_map.get(chain_id, 0))
        color = length_color.get(wallet_count, "#5e6472")
        plt.plot(grp["hop"], y, marker="o", markersize=4, linewidth=1.2, alpha=0.9, color=color)

    max_hop = int(details["hop"].max()) if not details.empty else 1
    plt.xticks(range(1, max_hop + 1))
    plt.title("Linear Peel-Chain Layout (One Horizontal Row per Chain)")
    plt.xlabel("Hop Number (1 = first transfer in chain)")
    plt.ylabel("Chain ID")
    plt.grid(alpha=0.25)

    legend_handles = [
        mpatches.Patch(color=length_color[4], label="4-wallet chains"),
        mpatches.Patch(color=length_color[5], label="5-wallet chains"),
        mpatches.Patch(color=length_color[6], label="6-wallet chains"),
    ]
    plt.legend(handles=legend_handles, title="Chain Length", frameon=False, loc="upper right")
    plt.tight_layout()
    plt.figtext(
        0.01,
        0.01,
        "X-axis: hop progression. Y-axis: chain index. Color encodes wallets per chain.",
        fontsize=9,
        alpha=0.8,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_chain_length_distribution(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    counts = summary["wallet_count"].value_counts().sort_index()
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(counts.index.astype(str), counts.values, color="#2a9d8f", alpha=0.9, label="Chain count")
    plt.title("Peel-Chain Wallet Length Distribution")
    plt.xlabel("Wallet Count per Chain")
    plt.ylabel("Number of Chains")
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


def _short_addr(addr: str, head: int = 8, tail: int = 6) -> str:
    addr = str(addr)
    if len(addr) <= head + tail + 3:
        return addr
    return f"{addr[:head]}...{addr[-tail:]}"


def _print_terminal_report(details: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 80)
    print("PEEL CHAIN REPORT")
    print("=" * 80)
    print(f"Total chains: {len(summary)}")
    print(f"Total peel transactions: {len(details)}")

    if summary.empty:
        print("No peel chains found.")
        return

    print(
        "Wallet length range: "
        f"{int(summary['wallet_count'].min())} to {int(summary['wallet_count'].max())}"
    )
    print(
        "Transaction length range: "
        f"{int(summary['tx_count'].min())} to {int(summary['tx_count'].max())}"
    )
    anomalies = summary[summary["anomaly"].astype(str) != ""]
    print(f"Chains with anomalies: {len(anomalies)}")

    length_counts = summary["wallet_count"].value_counts().sort_index()
    print("Length distribution (wallet_count -> chains):")
    for length, count in length_counts.items():
        print(f"  {int(length)} -> {int(count)}")

    print("-" * 80)
    print("CHAIN SUMMARY TABLE")
    print("-" * 80)
    header = (
        f"{'ID':>3} | {'W':>2} | {'TX':>2} | {'Start Amt':>11} | {'End Amt':>11} | "
        f"{'Total Drop%':>10} | {'Per-hop Drop%':>17} | {'Anomaly':<16}"
    )
    print(header)
    print("-" * len(header))

    for _, row in summary.sort_values("chain_id").iterrows():
        start_amount = row["start_amount"]
        end_amount = row["end_amount"]
        drop_total = "N/A"
        if pd.notna(start_amount) and pd.notna(end_amount) and start_amount not in (0,):
            drop_total = f"{((start_amount - end_amount) / start_amount) * 100:.2f}"

        hop_drop = "N/A"
        if pd.notna(row.get("min_drop_pct")) and pd.notna(row.get("max_drop_pct")):
            hop_drop = f"{float(row['min_drop_pct']):.2f} - {float(row['max_drop_pct']):.2f}"

        anomaly = str(row.get("anomaly", "")).strip() or "none"
        print(
            f"{int(row['chain_id']):>3} | {int(row['wallet_count']):>2} | {int(row['tx_count']):>2} | "
            f"{float(start_amount):>11.6f} | {float(end_amount):>11.6f} | "
            f"{drop_total:>10} | {hop_drop:>17} | {anomaly:<16}"
        )

    print("-" * 80)
    print("CHAIN DETAILS")
    print("-" * 80)

    for _, row in summary.sort_values("chain_id").iterrows():
        chain_id = int(row["chain_id"])
        chain_steps = details[details["chain_id"] == chain_id].sort_values("hop")

        start_amount = row["start_amount"]
        end_amount = row["end_amount"]
        drop_total = None
        if pd.notna(start_amount) and pd.notna(end_amount) and start_amount not in (0,):
            drop_total = ((start_amount - end_amount) / start_amount) * 100

        print(
            f"Chain {chain_id} | wallets={int(row['wallet_count'])} | tx={int(row['tx_count'])} | "
            f"start={_short_addr(row['start_wallet'])} | end={_short_addr(row['end_wallet'])}"
        )
        if drop_total is not None:
            print(
                f"  Amount decay: start={start_amount:.8f}, end={end_amount:.8f}, total_drop={drop_total:.2f}%"
            )
        if pd.notna(row.get("min_drop_pct")) and pd.notna(row.get("max_drop_pct")):
            print(
                f"  Per-hop drop range: {float(row['min_drop_pct']):.2f}% to {float(row['max_drop_pct']):.2f}%"
            )
        if str(row.get("anomaly", "")):
            print(f"  Anomaly: {row['anomaly']}")

        for _, step in chain_steps.iterrows():
            drop_val = step.get("drop_pct_from_prev")
            drop_txt = "N/A" if pd.isna(drop_val) else f"{float(drop_val):.2f}%"
            print(
                f"    hop {int(step['hop'])}: {_short_addr(step['from'])} -> {_short_addr(step['to'])} | "
                f"amount={float(step['amount']):.8f} | drop_from_prev={drop_txt}"
            )

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal-first peel-chain report with image outputs in one subfolder.")
    parser.add_argument("--transactions", default="data/processed/augmented_transactions_peel.csv")
    parser.add_argument("--labels", default="data/processed/transaction_labels_peel.csv")
    parser.add_argument("--out-dir", default="reports/Peel Chain Visualization")
    args = parser.parse_args()

    out_dir = _resolve_user_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions_path = _resolve_user_path(args.transactions)
    labels_path = _resolve_user_path(args.labels) if args.labels else None

    peel_df = _load_peel_rows(transactions_path, labels_path)
    details, summary = _build_chains(peel_df)

    _print_terminal_report(details, summary)

    profile_plot = out_dir / "peel_chain_amount_profiles.png"
    layout_plot = out_dir / "peel_chain_linear_layout.png"
    length_plot = out_dir / "peel_chain_length_distribution.png"
    _plot_chain_amount_profiles(details, profile_plot)
    _plot_linear_chain_layout(details, summary, layout_plot)
    _plot_chain_length_distribution(summary, length_plot)

    print("=" * 80)
    print("IMAGES SAVED")
    print("=" * 80)
    print(profile_plot)
    print(layout_plot)
    print(length_plot)


if __name__ == "__main__":
    main()
