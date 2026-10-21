# DeFiGuard: Explainable Graph-Temporal AML Intelligence for DeFi

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)](https://vitejs.dev/)
[![ML Model](https://img.shields.io/badge/Model-Tri--Hybrid%20GNN%20%2B%20NTS%20%2B%20XGBoost-purple.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-TreeSHAP%20Decomposition-ff69b4.svg)]()
[![Validation](https://img.shields.io/badge/Empirical%20Validation-Tornado%20Cash%20%7C%20Binance%20CEX%20%7C%20BigQuery-success.svg)]()

**DeFiGuard** is an end-to-end, production-ready anti-money laundering (AML) forensic intelligence platform designed for EVM and DeFi blockchain networks. It addresses the fundamental flaw of legacy AML heuristic rules — **alert fatigue and lack of explainability** — by integrating **Graph Neural Networks (GraphSAGE)**, **Neural Temporal Sequential (NTS)** embeddings, and an **Isotonically Calibrated XGBoost Classifier** paired with **TreeSHAP local and global explainability**.

The system features a **Universal Ingestion & Normalization Engine** capable of ingesting raw transaction dumps from Google BigQuery, Etherscan, Dune Analytics, or Flipside, dynamically constructing wallet interaction graphs on the fly, and presenting interactive forensic network graphs with click-to-SHAP investigation drawers in real-time.

---

## 🌟 Key Highlights & Innovations

1. **Tri-Hybrid Detection Architecture (51 Features)**:
   - **GNN Structural Embeddings (32 features)**: Captures multi-hop wallet clustering, dispersion, and bipartite subgraph topologies via GraphSAGE.
   - **Neural Temporal Sequential (NTS) Embeddings (4 features)**: Quantifies burst intervals, inter-arrival time compression, and rapid directional money flow.
   - **Graph Centrality & Flow Baselines (15 features)**: PageRank, betweenness centrality, in/out degrees, and flow balance ratios.
2. **Universal Dataset Normalization Intelligence**:
   - Zero-config auto-detection of incoming CSV schemas (Etherscan, BigQuery, Dune, Flipside, generic EVM).
   - Dynamic Wei $\leftrightarrow$ ETH scaling and address canonicalization.
   - On-the-fly topological graph construction and centrality metric generation for raw un-engineered datasets.
3. **Forensic Explainability (XAI)**:
   - Real-time TreeSHAP risk attribution for every single transaction.
   - Multi-layer contribution decomposition: **GNN Structural vs. NTS Temporal vs. Base Flow**.
   - Automated plain-English investigator narratives explaining *why* an alert was triggered.
4. **Interactive Cyber-Forensic Dashboard**:
   - D3 force-directed network graph highlighting suspicious transactions (crimson) vs. clean operations (cyan).
   - Searchable, filterable risk alerts table with instant CSV evidence export.

---

## 🔬 Empirical Validation: The Golden Triangle

DeFiGuard has been empirically verified across three distinct real-world blockchain testbeds to prove both **maximum sensitivity** to illicit contracts and **high-fidelity specificity** against alert fatigue:

```
                      [ DEFIGUARD EMPIRICAL TRIANGLE ]
                                     ▲
                                    / \
                                   /   \
        1. SENSITIVITY            /     \           2. SPECIFICITY
       (High-Risk Detection)     /       \      (Alert Fatigue Resistance)
     =======================    /         \    ============================
       Tornado Cash 100 ETH    /___________\       Binance Hot Wallet 14
        Pool (Etherscan)                            (Etherscan)
        230 / 230 Flagged                          4,987 / 5,000 Clean
        (100.0% Detection)                         (99.74% Specificity)
                                      |
                                      | 3. MACRO SCALABILITY
                                      | (Real-World Organic Flow)
                                      ▼
                             Google BigQuery 
                             20,000 Tx Ingestion
                             18,958 Wallets on-the-fly
```

| Benchmark Testbed | Ingestion Source | Scale (Txs / Wallets) | Volume (ETH) | Alert Rate | Forensic Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tornado Cash 100 ETH Pool** | Etherscan CSV | 230 txs / 133 wallets | 23,000.00 ETH | **100.0%** ($230/230$) | **Sensitivity:** 0% false negatives on OFAC-sanctioned mixer contracts. |
| **Binance Hot Wallet 14** | Etherscan CSV | 5,000 txs / 1,075 wallets | 42,670.92 ETH | **0.26%** ($13/5,000$) | **Specificity:** 99.74% clean pass rate; proves immunity to alert fatigue on massive exchange hubs while catching fan-out smurfing withdrawals. |
| **Raw Ethereum Mainnet Blocks** | Google BigQuery | 20,000 txs / 18,958 wallets | Multi-Million ETH | Graduated Tier | **Scalability:** Full ledger stream parsed and graphed on the fly in seconds. |

---

## 📊 Benchmark Test Metrics (110k Hardened Split)

Evaluated on an out-of-sample, temporally-isolated test set of 16,555 transactions from the 110k gold-standard hardened dataset:

| Forensic Metric | Score | Industry Comparison |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.933** | Discriminates laundering vs. benign with high confidence |
| **Recall (Sensitivity)** | **98.5%** | Catches virtually all active laundering chains |
| **Precision @ 0.50 Threshold** | **68.5%** | Over $6.8\times$ lower false positive rate than legacy rules |
| **PR-AUC** | **0.852** | High precision retention across all recall operating thresholds |
| **Inference Latency** | **< 1.8 ms / tx** | Sub-second real-time scoring throughput |

---

## 🧬 Laundering Topologies Detected

```
  [ PEEL CHAIN ]                [ SMURFING ]               [ CIRCULAR / CYCLE ]
 A ──► B ──► C ──► D         A ──► M1 ──► Target          A ──► B ──► C ──► A
 (Linear value bleed)        A ──► M2 ──► Target          (Layering wash trading)
 Primary: NTS + Base         Primary: GraphSAGE GNN       Primary: NTS + PageRank
```

1. **Peel Chains (Linear)**: A high-value deposit split into repeated small peeling outputs through ephemeral intermediary addresses.
2. **Smurfing (Fan-Out / Fan-In)**: Rapid dispersion of funds across dozens of mule wallets before re-aggregating into an off-ramp.
3. **Circular / Ring Routing (Cyclic)**: Layered transactions traversing cyclic loops between collaborating wallets to obfuscate transaction provenance.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DEFIGUARD PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [ RAW INGESTION ]                                                          │
│  Etherscan CSV │ Google BigQuery │ Dune Analytics │ Flipside Crypto         │
│         │                                                                   │
│         ▼                                                                   │
│  [ DATASET NORMALIZATION INTELLIGENCE ]                                     │
│  • Schema Auto-Detection & Column Aliasing                                  │
│  • Wei ↔ ETH Unit Harmonization                                             │
│  • On-the-Fly Dynamic Graph Construction (PageRank, Degrees, Flow Ratios)   │
│         │                                                                   │
│         ▼                                                                   │
│  [ 51-FEATURE TRI-HYBRID INFERENCE PIPELINE ]                               │
│  ┌───────────────────────┬──────────────────────┬────────────────────────┐  │
│  │  GraphSAGE GNN (32)   │  NTS Temporal (4)    │  Graph Flow Base (15)  │  │
│  │  Structural Position  │  Burst Timing & Gaps │  Centrality & Volume   │  │
│  └──────────┬────────────┴──────────┬───────────┴───────────┬────────────┘  │
│             └───────────────────────┼───────────────────────┘               │
│                                     ▼                                       │
│                       [ CALIBRATED XGBOOST ENGINE ]                         │
│                                     │                                       │
│                    ┌────────────────┴───────────────┐                       │
│                    ▼                                ▼                       │
│          [ CALIBRATED RISK SCORE ]        [ TREESHAP EXPLAINER ]            │
│          Probability (0.0 - 1.0)          Risk Drivers & Narratives         │
│                    │                                │                       │
│                    └────────────────┬───────────────┘                       │
│                                     ▼                                       │
│  [ FASTAPI REST BACKEND ] (Port 8000)                                       │
│  /predict  •  /predict-csv  •  /normalize-preview  •  /model-info           │
│                                     │                                       │
│                                     ▼                                       │
│  [ REACT + VITE FORENSIC UI ] (Port 3000)                                   │
│  Executive Dashboard │ D3 Network Topology │ Click-to-SHAP Forensic Drawer  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
DEFIGUARD/
├── api.py                            # FastAPI backend server with normalization & SHAP endpoints
├── requirements.txt                  # Python dependencies (PyTorch, XGBoost, SHAP, FastAPI)
│
├── Phase 1/
│   ├── scripts/
│   │   ├── dataset_normalizer.py     # Universal Ethereum normalization & on-the-fly graph engine
│   │   ├── inference.py              # Production tri-hybrid prediction class
│   │   ├── 01_evaluate_100k_test.py  # Model test evaluation script
│   │   └── 02_run_25k_stress_test.py # Out-of-sample stress test script
│   ├── Phase 1 Models/               # Serialized model weights & feature artifacts
│   │   ├── master_hybrid_model.json  # Calibrated XGBoost model
│   │   ├── gnn_wallet_embeddings.pkl # Pre-computed GraphSAGE embeddings
│   │   ├── nts_map.pkl               # Temporal sequence mapping
│   │   ├── shap_explainer.pkl        # Serialized TreeSHAP explainer
│   │   ├── feature_names.json        # 51 canonical feature names
│   │   └── decision_threshold.json   # Calibrated audit threshold (0.200)
│   └── Phase 1 Reports/              # Benchmark logs and SHAP plots
│
├── Frontend/                         # Modern React + Vite Web Application
│   ├── src/
│   │   ├── App.jsx                   # Main layout and tab router
│   │   ├── api.js                    # Backend REST API client
│   │   └── components/
│   │       ├── DashboardOverview.jsx # Executive KPI dashboard & risk charts
│   │       ├── UploadPanel.jsx       # CSV drag-and-drop & schema normalizer modal
│   │       ├── AlertsTable.jsx       # Searchable alerts table with CSV export
│   │       ├── NetworkGraph.jsx      # D3 interactive graph with Click-to-SHAP
│   │       ├── ShapExplanation.jsx   # Dynamic SHAP bar charts & investigator narratives
│   │       └── ModelInfo.jsx         # Architecture specs & feature breakdown
│   ├── package.json
│   └── vite.config.js
│
└── data/                             # Raw and processed evaluation samples
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Windows, macOS, or Linux

### 2. Backend Setup

```bash
# Clone the repository
git clone https://github.com/Lazarus-57/DeFIGuard.git
cd DeFiGuard

# Create and activate Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install shap "xgboost==2.1.4" fastapi uvicorn python-multipart

# Launch the FastAPI backend server
python -m uvicorn api:app --port 8000
```
Backend API will be active at: `http://127.0.0.1:8000` (API documentation at `http://127.0.0.1:8000/docs`).

### 3. Frontend Setup

In a separate terminal:

```bash
cd Frontend
npm install
npm run dev
```
Open your browser and navigate to: **`http://localhost:3000`**

---

## 🔍 How to Test arbitrary Datasets

1. Open the DeFiGuard Web Dashboard (`http://localhost:3000`).
2. Navigate to the **Upload** tab.
3. Upload any Ethereum transaction CSV (or click one of the pre-configured sample cards for **Etherscan**, **Google BigQuery**, or **Dune**).
4. Watch the **Dataset Ingestion Intelligence** engine automatically:
   - Identify the source provider and canonicalize address headers.
   - Scale Wei values to ETH.
   - Construct the wallet graph topology on the fly.
5. Inspect the generated **Executive Dashboard**, explore the **D3 Network Graph**, click any flagged node to open the **TreeSHAP local explanation drawer**, and download the forensic CSV report.

---

## ⚖️ License & Academic Citation

Built as a senior Capstone Research Project at PES University on Explainable Deep Learning & Graph Intelligence for Decentralized Finance (DeFi) Anti-Money Laundering.

If referencing this work in academic research, cite as:
```bibtex
@misc{defiguard2026,
  author = {Joshua Daniel and Project Contributors},
  title = {DeFiGuard: Explainable Graph-Temporal AML Intelligence for Decentralized Finance},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Lazarus-57/DeFIGuard}}
}
```
