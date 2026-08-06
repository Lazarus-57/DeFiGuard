import random

import pandas as pd


def _random_eth_address(rng: random.Random) -> str:
    return f"0x{rng.getrandbits(160):040x}"


def generate_peel_chains(
    num_chains: int,
    existing_wallets: set[str] | None = None,
    chain_length_weights: dict[int, float] | None = None,
    seed: int | None = 42,
    real_amounts: list[float] | None = None,
    anchor_wallets: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic peel-chain transactions.

    Each chain has 4-6 wallets and 3-5 transactions where each transfer
    amount decreases by 2-5% from the previous transfer.

    If ``real_amounts`` is provided, the starting amount for each chain is
    sampled from that list (non-zero values only) so synthetic amounts
    follow the same distribution as real transaction amounts.

    If ``anchor_wallets`` is provided, the first wallet of each chain is
    sampled from real existing wallets so the peel chain connects to the
    real transaction graph instead of forming an isolated subgraph.
    Intermediate and final wallets remain fresh random addresses (realistic:
    launderers create new addresses for layering).
    """
    rows = []
    rng = random.Random(seed)
    used_wallets = set(existing_wallets or set())
    length_weights = chain_length_weights or {4: 0.2, 5: 0.5, 6: 0.3}
    chain_lengths = sorted(length_weights.keys())
    weights = [length_weights[length] for length in chain_lengths]

    # Pre-filter anchor wallets and real amounts for sampling.
    _anchor_list = list(anchor_wallets) if anchor_wallets else []
    _real_nonzero = [a for a in (real_amounts or []) if a > 0]

    for chain_idx in range(num_chains):
        # Favor medium chains (5-6 wallets) while keeping the required 4-6 range.
        chain_length = rng.choices(chain_lengths, weights=weights, k=1)[0]
        wallets = []

        # First wallet: anchor to a real wallet if available.
        if _anchor_list:
            wallets.append(rng.choice(_anchor_list))
        else:
            wallet = _random_eth_address(rng)
            while wallet in used_wallets:
                wallet = _random_eth_address(rng)
            wallets.append(wallet)
            used_wallets.add(wallet)

        # Remaining wallets: fresh random addresses.
        for _ in range(chain_length - 1):
            wallet = _random_eth_address(rng)
            while wallet in used_wallets:
                wallet = _random_eth_address(rng)
            wallets.append(wallet)
            used_wallets.add(wallet)

        # Starting amount: sample from real distribution if available.
        if _real_nonzero:
            amount = rng.choice(_real_nonzero)
        else:
            amount = rng.uniform(5.0, 50.0)

        for i in range(chain_length - 1):
            rows.append(
                {
                    "from": wallets[i],
                    "to": wallets[i + 1],
                    "amount": round(amount, 8),
                    "pattern": "peel_chain",
                    "pattern_id": f"peel_{chain_idx}",
                }
            )
            amount *= 1 - rng.uniform(0.02, 0.05)

    return pd.DataFrame(rows, columns=["from", "to", "amount", "pattern", "pattern_id"])


def generate_smurf_clusters(
    num_clusters: int,
    existing_wallets: set[str] | None = None,
    mule_count_weights: dict[int, float] | None = None,
    seed: int | None = 42,
    real_amounts: list[float] | None = None,
    anchor_wallets: list[str] | None = None,
    jitter_range: tuple[float, float] = (0.05, 0.15),
    forward_loss_range: tuple[float, float] = (0.01, 0.03),
) -> pd.DataFrame:
    """Generate synthetic smurfing-cluster transactions.

    Each cluster follows a fan-out / fan-in topology:

        Source ──→ Mule_1 ──→ Collector
               ──→ Mule_2 ──→ Collector
               ──→ ...    ──→ Collector
               ──→ Mule_N ──→ Collector

    The source wallet splits a total amount roughly equally across N mule
    wallets (with random jitter).  Each mule then forwards its received
    amount to a single collector wallet, minus a small fee/loss.

    If ``real_amounts`` is provided, the cluster's total amount is sampled
    from that list so synthetic amounts follow the real distribution.

    If ``anchor_wallets`` is provided, the source wallet of each cluster
    is sampled from real existing wallets so the smurfing cluster connects
    to the real transaction graph.

    Parameters
    ----------
    num_clusters : int
        Number of smurfing clusters to generate.
    existing_wallets : set[str] | None
        Already-used wallet addresses to avoid collisions.
    mule_count_weights : dict[int, float] | None
        Mapping of mule-count → sampling weight.  Defaults favour 4-6 mules.
    seed : int | None
        Random seed for reproducibility.
    real_amounts : list[float] | None
        Non-zero real transaction amounts to sample from.
    anchor_wallets : list[str] | None
        Real wallets for anchoring source addresses.
    jitter_range : tuple[float, float]
        (min, max) fractional jitter applied to equal-split amounts.
    forward_loss_range : tuple[float, float]
        (min, max) fractional fee each mule subtracts before forwarding.

    Returns
    -------
    pd.DataFrame
        Columns: ``from``, ``to``, ``amount``, ``pattern``.
        All rows have ``pattern == "smurfing"``.
    """
    rows: list[dict] = []
    rng = random.Random(seed)
    used_wallets = set(existing_wallets or set())
    weights_map = mule_count_weights or {3: 0.1, 4: 0.2, 5: 0.3, 6: 0.25, 7: 0.1, 8: 0.05}
    mule_counts = sorted(weights_map.keys())
    weights = [weights_map[k] for k in mule_counts]

    _anchor_list = list(anchor_wallets) if anchor_wallets else []
    _real_nonzero = [a for a in (real_amounts or []) if a > 0]

    for cluster_idx in range(num_clusters):
        num_mules = rng.choices(mule_counts, weights=weights, k=1)[0]

        # --- Source wallet (anchored to real wallet when possible) ---
        if _anchor_list:
            source = rng.choice(_anchor_list)
        else:
            source = _random_eth_address(rng)
            while source in used_wallets:
                source = _random_eth_address(rng)
            used_wallets.add(source)

        # --- Mule wallets (always fresh) ---
        mules: list[str] = []
        for _ in range(num_mules):
            mule = _random_eth_address(rng)
            while mule in used_wallets:
                mule = _random_eth_address(rng)
            mules.append(mule)
            used_wallets.add(mule)

        # --- Collector wallet (always fresh) ---
        collector = _random_eth_address(rng)
        while collector in used_wallets:
            collector = _random_eth_address(rng)
        used_wallets.add(collector)

        # --- Total amount for this cluster ---
        if _real_nonzero:
            total_amount = rng.choice(_real_nonzero)
        else:
            total_amount = rng.uniform(5.0, 50.0)

        # --- Fan-out: source → each mule (near-equal split with jitter) ---
        base_split = total_amount / num_mules
        mule_received: list[float] = []
        for mule in mules:
            jitter_frac = rng.uniform(*jitter_range)
            direction = rng.choice([-1, 1])
            split_amount = base_split * (1.0 + direction * jitter_frac)
            split_amount = max(split_amount, 0.0)
            mule_received.append(split_amount)
            rows.append({
                "from": source,
                "to": mule,
                "amount": round(split_amount, 8),
                "pattern": "smurfing",
                "pattern_id": f"smurf_{cluster_idx}",
            })

        # --- Fan-in: each mule → collector (minus forward loss) ---
        for mule, received in zip(mules, mule_received):
            loss_frac = rng.uniform(*forward_loss_range)
            forwarded = received * (1.0 - loss_frac)
            rows.append({
                "from": mule,
                "to": collector,
                "amount": round(forwarded, 8),
                "pattern": "smurfing",
                "pattern_id": f"smurf_{cluster_idx}",
            })

    return pd.DataFrame(rows, columns=["from", "to", "amount", "pattern", "pattern_id"])


def generate_circular_laundering(
    num_rings: int,
    existing_wallets: set[str],
    ring_size_weights: dict[int, float],
    seed: int = 42,
    real_amounts: list[float] | None = None,
    anchor_wallets: list[str] | None = None,
    decay_range: tuple[float, float] = (0.02, 0.05),
) -> pd.DataFrame:
    """Generates synthetic 'circular' or 'U-turn' laundering patterns.

    In a circular pattern, funds start at a source (anchor) wallet, move through
    a sequence of intermediary wallets, and eventually return to the source
    wallet. At each hop, a small 'fee' or 'decay' occurs to make the amounts
    unequal.

    Args:
        num_rings: Number of circular patterns to generate.
        existing_wallets: Set of known wallet addresses to avoid collisions.
        ring_size_weights: Weights for ring sizes (number of hops/nodes in cycle).
        seed: Random seed for reproducibility.
        real_amounts: Pool of actual amounts to sample from for realism.
        anchor_wallets: Pool of existing wallets to use as the source/target.
        decay_range: Range of percentage loss (0.0 to 1.0) per hop (e.g., fee).

    Returns:
        DataFrame of synthetic circular transactions (from, to, amount, pattern).
    """
    rng = random.Random(seed)
    used_wallets = set(existing_wallets)
    rows: list[dict[str, str | float]] = []

    ring_sizes = list(ring_size_weights.keys())
    weights = list(ring_size_weights.values())

    _anchor_list = anchor_wallets or []
    _real_nonzero = [a for a in (real_amounts or []) if a > 0]

    for ring_idx in range(num_rings):
        ring_size = rng.choices(ring_sizes, weights=weights, k=1)[0]
        if ring_size < 3:
            ring_size = 3  # Min 3 hops (A->B->C->A)

        # --- Source/Target wallet (anchored to real wallet when possible) ---
        if _anchor_list:
            source = rng.choice(_anchor_list)
        else:
            source = _random_eth_address(rng)
            while source in used_wallets:
                source = _random_eth_address(rng)
            used_wallets.add(source)

        # --- Intermediary wallets (always fresh) ---
        intermediaries: list[str] = []
        for _ in range(ring_size - 1):
            hop = _random_eth_address(rng)
            while hop in used_wallets:
                hop = _random_eth_address(rng)
            intermediaries.append(hop)
            used_wallets.add(hop)

        # --- Base amount ---
        if _real_nonzero:
            current_amount = rng.choice(_real_nonzero)
        else:
            current_amount = rng.uniform(10.0, 100.0)

        # --- Create hops ---
        path = [source] + intermediaries + [source]
        
        for i in range(len(path) - 1):
            sender = path[i]
            receiver = path[i + 1]
            
            rows.append({
                "from": sender,
                "to": receiver,
                "amount": round(current_amount, 8),
                "pattern": "circular",
                "pattern_id": f"circular_{ring_idx}",
            })
            
            # Decay for next hop
            decay = rng.uniform(*decay_range)
            current_amount = current_amount * (1.0 - decay)

    return pd.DataFrame(rows, columns=["from", "to", "amount", "pattern", "pattern_id"])
