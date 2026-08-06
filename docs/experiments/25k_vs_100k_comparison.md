# 25k vs 100k Pipeline Comparison

This document provides a comparative analysis between the earlier 25k dataset evaluation and the finalized, corrected 100k dataset evaluation. 

Both datasets were injected with Peel Chains and Smurfing clusters, allowing for a direct structural comparison. However, three major differences deeply impact the interpretation of these results: **Dataset Scale**, **Positive Base Rate**, and **Temporal Data Leakage**.

## High-Level Metrics Comparison (Peel + Smurf)

| Metric | 25k Dataset (Legacy) | 100k Dataset (Fixed) |
| :--- | :--- | :--- |
| **Total Transactions** | 32,220 | 107,220 |
| **Injected Positives** | 7,220 | 7,220 |
| **Positive Base Rate** | 22.4% | 6.7% |
| **Baseline PR-AUC** | ~0.8961 | 0.1563 |
| **GNN PR-AUC Lift** | -0.0679 (Max ~0.8282) | +0.0539 |
| **NTS PR-AUC Lift** | -0.0297 (Max ~0.8664) | +0.1064 |

---

## Key Insights & Differences

### 1. The "Base Rate" Illusion
The most striking difference between the two runs is the Baseline PR-AUC: **0.8961** in the 25k dataset versus **0.1563** in the 100k dataset. 

This massive drop is not a failure of the model, but rather a reflection of the **Base Rate Fallacy** in classification metrics. Because we injected the exact same number of synthetic patterns (7,220 transactions) into both datasets, the 25k dataset became heavily saturated with illicit activity (22.4% positive rate). The 100k dataset diluted this back to a realistic 6.7%. 
*   **Insight:** PR-AUC is highly sensitive to class imbalance. The 100k results provide a much more realistic picture of how difficult it is to detect these patterns in a massive sea of legitimate DeFi noise.

### 2. Temporal Data Leakage Impact
In the legacy 25k run, the synthetic timestamps were assigned completely randomly across the dataset's entire time range. This created a highly artificial "maximum spread" temporal signature. 
*   **25k Run (Leaky):** The Node Temporal Spread (NTS) feature actually *hurt* performance (-0.0297) because the random spread didn't map to any coherent illicit behavior; it just confused the model.
*   **100k Run (Fixed):** After we fixed the timestamp logic to force chronological sequence (e.g., batched fan-outs in a 30-second window), the NTS feature provided a massive **+0.1064** lift. 
*   **Insight:** True temporal intelligence requires realistic, chronologically accurate data. When launderers are forced to execute coordinated attacks quickly, NTS is our strongest weapon.

### 3. GraphSAGE (GNN) Scaling
In the 25k dataset, the GNN embeddings (GraphSAGE) failed to beat the baseline (dropping performance by ~0.06). However, in the 100k dataset, the GNN provided a solid **+0.0539** lift.
*   **Why?** Graph Neural Networks require massive, interconnected graphs to learn meaningful structural representations. The 25k dataset was simply too small and disjointed. In the 100k dataset, the broader context of legitimate DeFi liquidity pools, DEX routers, and arbitrage bots allowed GraphSAGE to effectively contrast the rigid structural topology of Smurfing clusters against normal ecosystem behavior.

---

## Conclusion
The 25k dataset served as an excellent functional sandbox to verify our pipeline architecture. However, due to the high base rate and the temporal data leakage, its metrics should be considered deprecated.

The **100k corrected results are the true, mathematically sound foundation** for this project moving forward. They prove that while Baseline XGBoost handles raw features well, adding GNNs (for structural cluster detection) and NTS (for temporal burst detection) provides critical, orthogonal lifts in performance in a realistic DeFi environment.
