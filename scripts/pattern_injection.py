import random

import pandas as pd


def _random_eth_address(rng: random.Random) -> str:
    return f"0x{rng.getrandbits(160):040x}"


def generate_peel_chains(
    num_chains: int,
    existing_wallets: set[str] | None = None,
    chain_length_weights: dict[int, float] | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synthetic peel-chain transactions.

    Each chain has 4-6 wallets and 3-5 transactions where each transfer
    amount decreases by 2-5% from the previous transfer.
    """
    rows = []
    rng = random.Random(seed)
    used_wallets = set(existing_wallets or set())
    length_weights = chain_length_weights or {4: 0.2, 5: 0.5, 6: 0.3}
    chain_lengths = sorted(length_weights.keys())
    weights = [length_weights[length] for length in chain_lengths]

    for _ in range(num_chains):
        # Favor medium chains (5-6 wallets) while keeping the required 4-6 range.
        chain_length = rng.choices(chain_lengths, weights=weights, k=1)[0]
        wallets = []
        for _ in range(chain_length):
            wallet = _random_eth_address(rng)
            while wallet in used_wallets:
                wallet = _random_eth_address(rng)
            wallets.append(wallet)
            used_wallets.add(wallet)

        amount = rng.uniform(5.0, 50.0)
        for i in range(chain_length - 1):
            rows.append(
                {
                    "from": wallets[i],
                    "to": wallets[i + 1],
                    "amount": round(amount, 8),
                    "pattern": "peel_chain",
                }
            )
            amount *= 1 - rng.uniform(0.02, 0.05)

    return pd.DataFrame(rows, columns=["from", "to", "amount", "pattern"])
