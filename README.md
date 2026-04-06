# DeFIGuard: A Framework for Analyzing Money Laundering in DeFi Transactions

DeFIGuard is a research-oriented pipeline for detecting suspicious fund flows in blockchain transaction graphs, with emphasis on temporal laundering behavior and leakage-safe evaluation.

## Why This Project

DeFi money laundering often uses multi-hop movement, wallet fan-out, and timing obfuscation. Static, single-transaction signals are usually not enough. DeFIGuard combines transaction, graph, and temporal features to improve detection quality on imbalanced AML data.

## Core Objectives

- Build a reproducible AML detection pipeline from raw chain data to evaluation.
- Compare strong tabular baselines against graph-augmented and temporal-augmented models.
- Prioritize PR-AUC and recall-centric metrics appropriate for rare-event detection.
- Keep evaluation leakage-safe: train/val for model development, test for final holdout.

## Current Scope

- Chain focus: Ethereum (single-chain phase).
- Pattern work completed: peel-chain injection and evaluation.
- Next roadmap: scaling dataset size and adding smurfing/circular patterns.

## System Architecture

1. Data Collection
2. Graph & Pattern Pipeline
3. Modeling Table Prep
4. Layered Model Evaluation
5. Reporting / Visualization

## Detection Layers

### Layer 1: Baselines

- Logistic Regression
- Random Forest
- XGBoost

Goal: establish a strong tabular benchmark under class imbalance.

### Layer 2: Graph Context

- Standalone XGBoost
- XGBoost + GraphSAGE embeddings

Goal: test if wallet-graph representation improves ranking of suspicious transactions.

### Layer 3: Temporal Intelligence

- Tuned XGBoost baseline
- Tuned XGBoost + NTS (Network Time Spread features)

Goal: capture timing/sequence asymmetries typical in laundering behavior.

## Key Metrics

- PR-AUC (primary)
- ROC-AUC
- Precision
- Recall
- F1
- Recall@P80

PR-AUC is prioritized because AML labels are sparse.

## Data and Leakage Safety

- Temporal split: train 70%, val 15%, test 15%.
- Model/hyperparameter decisions use train + val only.
- Test set is preserved for final, one-time reporting.
- Feature maps for graph/temporal stats are fit on training split to reduce leakage risk.

## Repository Layout

```text
DEFIGUARD/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── reports/
├── scripts/
│   ├── download_dune.py
│   ├── graph_pipeline.py
│   ├── model_prep.py
│   ├── pattern_injection.py
│   ├── peel_chain_report.py
│   ├── phase1_model_selection.py
│   ├── phase1_step2_xgb_gnn_comparison.py
│   ├── phase1_step3_layer3_comparison.py
│   └── visualize_peel_chains.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### 1) Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure API key

```bash
$env:DUNE_API_KEY="<your_key>"
```

### 3) Pull raw data

```bash
python scripts/download_dune.py --query-id 6702728 --output eth_transfers_2024_01_01_1hr.csv
```

### 4) Build augmented + prepared datasets

```bash
python scripts/graph_pipeline.py
python scripts/model_prep.py
```

### 5) Run experiments

```bash
python scripts/phase1_model_selection.py
python scripts/phase1_step2_xgb_gnn_comparison.py
python scripts/phase1_step3_layer3_comparison.py --nts-mode spread_abs
```

## Development Workflow

- `main` branch: stable, reportable state.
- feature branches: scoped work (example: `model-scaling`).
- commit style: concise and meaningful (what changed + why).

## Security and Repo Hygiene

- No credentials are hardcoded.
- Large data artifacts are excluded from version control.
- Local virtual environments and IDE state are excluded.

See [data/README.md](data/README.md) for dataset handling and download instructions.

## Roadmap

1. Scale to 20K then 50K+ Ethereum transactions.
2. Add smurfing and circular flow patterns.
3. Integrate real-world illicit labels (sanctions/scam/rug lists).
4. Add explainability outputs (SHAP/path-level analysis).

## License

Add your preferred license (MIT/Apache-2.0) before public release.

## Acknowledgement

Built as a capstone framework for explainable AML analytics in DeFi transaction ecosystems.
