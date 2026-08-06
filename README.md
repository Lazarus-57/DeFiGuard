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
│   │   ├── 01_evaluate_100k_test.py # ← Run this: blind-test evaluation
│   │   ├── 02_run_25k_stress_test.py # ← Optional out-of-sample stress test
│   │   └── inference.py             # ← Backend API: call predict(df) to run the model
│   ├── Advanced/
│   │   └── phase1_master_training.py # ← Rebuilds artifacts; do not run for normal use
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
│   └── Explanation/                # ← Walkthrough and Phase 1 handoff guide
│
├── scripts/                        # ← Historical pipeline and experiment code
│   ├── pipeline/                   # Data collection, feature engineering, and pattern injection
│   ├── analysis/                   # Dataset checks, visualisation, and diagnostic reports
│   ├── experiments/                # Earlier dataset-scale experiment history
│       ├── 5k/                     # Prototype experiments
│       ├── 25k/                    # Architecture-validation experiments
│   └── 100k/                       # Gold-standard experiments (used by run_rerun.py)
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
├── docs/                           # Project documentation and exported reports
│   ├── experiments/                # 25k/100k experiment summaries
│   └── reports/                    # Capstone PDF reports
│
├── tools/                          # Utilities for generating project reports
│
├── requirements.txt
└── README.md
```

---

## Run Phase 1 Results (Start Here)

Use this section to verify the completed Phase 1 model. It evaluates the saved model on the untouched 100k test split; it does **not** retrain or overwrite the model.

### 1. Create the tested Python environment

Run these commands from the `DEFIGUARD` folder in PowerShell:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install shap "xgboost==2.1.4"
```

> If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once, then activate the environment again.

### 2. Run the blind-test evaluation

```powershell
python ".\Phase 1\scripts\01_evaluate_100k_test.py"
```

Expected headline metrics:

```text
Test Transactions: 16,555
ROC-AUC: 0.9330
PR-AUC: 0.4694
Precision: 0.4427
Recall: 0.9686
F1 Score: 0.6076
```

This is the final Phase 1 test: the model was trained on the training split, its threshold was selected on the validation split, and these 16,555 test transactions were held back for final evaluation.

### 3. Optional: run the 25k stress test

```powershell
python ".\Phase 1\scripts\02_run_25k_stress_test.py"
```

This is an out-of-distribution stress test on the legacy 25k dataset. It is useful for demonstrating temporal anomaly detection, but it is not the primary accuracy evaluation.

### Do not run this during normal use

`Phase 1/Advanced/phase1_master_training.py` retrains the model and overwrites the stored Phase 1 artifacts. Run it only when deliberately reproducing the full training pipeline.

---

## Use the Inference API (Phase 2 Developers)

The entire ML pipeline is wrapped in a single callable class. You do not need to understand GNNs or NTS to use it.

```python
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Phase 1/scripts").resolve()))
from inference import DefiGuardInference

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

See `Phase 1/Explanation/walkthrough.md` for the full handoff guide.

---

## Advanced: Reproduce Training

Only use this workflow when you intentionally want to regenerate the serialized model, embeddings, NTS map, SHAP explainer, reports, and metrics:

```powershell
python ".\Phase 1\Advanced\phase1_master_training.py"
```

This can take substantial time and overwrites the existing Phase 1 artifacts. For normal verification, use `01_evaluate_100k_test.py` instead.

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

None

## Acknowledgement

Built as a Capstone project for explainable AML analytics in DeFi transaction ecosystems.
