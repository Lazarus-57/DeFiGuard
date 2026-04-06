import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


INPUT_FILE = "augmented_transactions_peel.csv"
LABELS_FILE = "transaction_labels_peel.csv"
SUMMARY_FILE = "peel_chain_summary.csv"
PLOT_FILE = "peel_chains_overview.png"


def _extract_chains(peel_df: pd.DataFrame) -> list[list[str]]:
    edges = list(zip(peel_df["from"], peel_df["to"]))
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    from_set = set(peel_df["from"])
    to_set = set(peel_df["to"])
    starts = sorted(from_set - to_set)

    chains: list[list[str]] = []
    next_map = {src: dst for src, dst in edges}
    for start in starts:
        chain = [start]
        current = start
        while current in next_map:
            current = next_map[current]
            chain.append(current)
        chains.append(chain)

    return chains


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    if "aml_label" in df.columns:
        peel_df = df[df["aml_label"] == 1].copy()
    elif "tx_hash" in df.columns:
        labels = pd.read_csv(LABELS_FILE)
        peel_hashes = set(labels.loc[labels["aml_label"] == 1, "tx_hash"])
        peel_df = df[df["tx_hash"].isin(peel_hashes)].copy()
    else:
        peel_df = df[df["pattern"] == "peel_chain"].copy()

    if peel_df.empty:
        print("No peel_chain rows found.")
        return

    chains = _extract_chains(peel_df)

    summary_rows = []
    for idx, chain in enumerate(chains, start=1):
        summary_rows.append(
            {
                "chain_id": idx,
                "wallet_count": len(chain),
                "tx_count": len(chain) - 1,
                "start_wallet": chain[0],
                "end_wallet": chain[-1],
                "path": " -> ".join(chain),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["wallet_count", "chain_id"], ascending=[False, True])
    summary_df.to_csv(SUMMARY_FILE, index=False)

    graph = nx.DiGraph()
    graph.add_edges_from(zip(peel_df["from"], peel_df["to"]))

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(graph, seed=42, k=0.65)
    nx.draw_networkx_nodes(graph, pos, node_size=90, node_color="#1f77b4", alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color="#ff7f0e", arrows=True, arrowsize=8, width=1.2, alpha=0.8)

    # Label only chain start nodes to keep the plot readable.
    starts = sorted(set(peel_df["from"]) - set(peel_df["to"]))
    label_map = {node: f"S{i+1}" for i, node in enumerate(starts)}
    nx.draw_networkx_labels(graph, pos, labels=label_map, font_size=7)

    plt.title("Peel-Chain Subgraph (Start Nodes Labeled as S1, S2, ...)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=220)
    plt.close()

    counts = summary_df["wallet_count"].value_counts().sort_index()
    print(f"Total peel chains: {len(summary_df)}")
    for length, count in counts.items():
        print(f"Wallet length {int(length)}: {int(count)} chains")
    print(f"Saved chain summary: {SUMMARY_FILE}")
    print(f"Saved overview plot: {PLOT_FILE}")


if __name__ == "__main__":
    main()
