# DeFIGuard: ML-Powered Anti-Money Laundering for DeFi Transactions

[![Phase](https://img.shields.io/badge/Phase-1%20Complete-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-100k%20Transactions-blue)]()
[![Model](https://img.shields.io/badge/Model-Hybrid%20XGBoost%20%2B%20GNN%20%2B%20NTS-purple)]()
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.933-orange)]()
[![Recall](https://img.shields.io/badge/Recall-96.8%25-red)]()

DeFIGuard is a production-ready machine learning pipeline for detecting money laundering patterns in DeFi blockchain transactions. It combines **Graph Neural Networks (GNN)**, **Temporal Intelligence (NTS)**, and **XGBoost** into a single hybrid detection engine with full **SHAP-based explainability** — capable of flagging suspicious wallets, classifying the laundering pattern type, and explaining *why* the transaction was flagged.

---

## Current Status: Phase 1 Complete ✅

All machine learning work is complete. The trained model and its inference API are ready for backend/frontend integration (Phase 2).

### Final Test Set Metrics (100k Gold Standard Dataset)

| Metric | Score |
|---|---|
| ROC-AUC | **0.933** |
| Recall (sensitivity) | **96.8%** |
| Precision | 44.2% |
| F1 Score | 60.8% |
| PR-AUC | 0.469 |
| Decision Threshold | 0.448 |

> **Recall of 96.8% means the model catches nearly every single money laundering transaction in the test set.** At a positive base rate of ~9%, a precision of 44% means investigators spend nearly half their time on genuine alerts — far superior to legacy rule-based systems.

---

## Laundering Patterns Detected

| Pattern | Topology | Primary Detector |
|---|---|---|
| **Peel Chain** | Linear (A→B→C→D...) | XGBoost Baseline + NTS |
| **Smurfing** | Fan-out/Fan-in (1→N→1) | XGBoost + GraphSAGE GNN |
| **Circular Ring** | Cyclic (A→B→C→A) | XGBoost + NTS |

---

## Architecture: The Hybrid Model

The final model fuses three detection layers into a single 51-feature XGBoost classifier:

```
Raw Transactions
       │
       ├── Baseline Features (15)  ──────────────────────────────────┐
       │   amount, centrality, degree, flow_ratio...                  │
       │                                                              ▼
       ├── GNN Structural Embeddings (32) ─── GraphSAGE ──► 51-Feature ──► XGBoost ──► Suspicion Score
       │   wallet graph position, neighbourhood...                    ▲       │              │
       │                                                              │       │         SHAP Explainer
       └── NTS Temporal Features (4) ────────────────────────────────┘       │              │
           from_nts, to_nts, nts_max, nts_mean...                            │              ▼
                                                                              └──► Pattern Type Label
```

**Key Design Decisions:**
- **GNNs trained on train split only** — no data leakage into val/test
- **NTS computed on train split only** — enforces temporal integrity
- **Threshold tuned on val set**, applied once to test set

---

## Repository Layout

```
DEFIGUARD/
├── Phase 1/                        # ← The complete Phase 1 ML deliverable
│   ├── scripts/
│   │   ├── inference.py            # ← Backend API: call predict(df) to run the model
│   │   ├── phase1_master_training.py  # ← Master training script (re-runnable)
│   │   └── phase1_25k_inference_test.py  # ← Out-of-sample stress test
│   ├── Phase 1 Models/             # ← All serialized model artifacts
│   │   ├── master_hybrid_model.json
│   │   ├── gnn_wallet_embeddings.pkl
│   │   ├── nts_map.pkl
│   │   ├── shap_explainer.pkl
│   │   ├── feature_names.json
│   │   └── decision_threshold.json
│   ├── Phase 1 Reports/            # ← Final metrics and SHAP plots
│   │   ├── test_metrics.json
│   │   ├── shap_summary.png
│   │   └── 25k_inference_results.csv
│   └── walkthrough.md              # ← Full Phase 1 explanation + handoff guide
│
├── scripts/                        # ← Core pipeline scripts (shared across datasets)
│   ├── download_dune.py            # Data collection from Dune Analytics
│   ├── graph_pipeline.py           # Graph feature engineering
│   ├── model_prep.py               # Train/val/test split + feature table prep
│   ├── pattern_injection.py        # Synthetic laundering pattern generator
│   ├── 5k/                         # 5k prototype experiments
│   ├── 25k/                        # 25k architecture validation experiments
│   └── 100k/                       # 100k gold-standard experiments
│
├── data/
│   ├── raw/                        # Raw Ethereum transaction CSVs
│   └── processed/                  # Feature-engineered modeling datasets
│       ├── 5k/
│       ├── 25k/
│       └── 100k/
│
├── reports/                        # Summary reports and visualizations
│   ├── 25k/
│   └── 100k/
│
├── requirements.txt
└── README.md
```

---

## Quick Start: Using the Inference API (Phase 2 Developers)

The entire ML pipeline is wrapped in a single callable class. You do not need to understand GNNs or NTS to use it.

```python
import pandas as pd
from Phase_1.scripts.inference import DefiGuardInference

# Load once at application startup
ml_engine = DefiGuardInference()

# Pass any raw transaction DataFrame
raw_df = pd.read_csv("user_uploaded_transactions.csv")
results = ml_engine.predict(raw_df)

# results DataFrame now has these extra columns:
# - suspicion_score  (float 0–1)
# - aml_flag         (0 = clean, 1 = suspicious)
# - pattern_type     ("Normal", "Smurfing (Structural)", "Peel/Circular (Temporal)")
# - top_shap_reason  (the primary feature driving the alert)
```

See `Phase 1/walkthrough.md` for the full handoff guide.

---

## Reproducing the Results

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Re-run Master Training

```bash
python "Phase 1/scripts/phase1_master_training.py"
```

### 3. Run 25k Out-of-Sample Stress Test

```bash
python "Phase 1/scripts/phase1_25k_inference_test.py"
```

---

## Project Evolution

| Phase | Dataset | Focus | Status |
|---|---|---|---|
| Prototype | 5k txns | Peel Chain only, baseline models | ✅ Done |
| Validation | 25k txns | Add Smurfing, fix temporal leakage bug | ✅ Done |
| Gold Standard | 100k txns | All 3 patterns, full hybrid model | ✅ Done |
| **Phase 1** | **100k txns** | **Master model, SHAP, inference API** | ✅ **Complete** |
| Phase 2 | — | Backend API + Frontend Visualization | 🔜 Next |

---

## Roadmap: Phase 2

1. **Backend API** — Wrap `inference.py` in FastAPI/Flask with file-upload endpoint
2. **Graph Visualization** — Integrate Graphistry or PyVis to render flagged transaction networks
3. **Frontend UI** — Dashboard for uploading CSVs, viewing alerts, and inspecting SHAP explanations
4. **Real-World Labels** — Integrate sanctions/scam lists for ground truth enrichment

---

## License

Add your preferred license (MIT/Apache-2.0) before public release.

## Acknowledgement

Built as a Capstone project for explainable AML analytics in DeFi transaction ecosystems.
