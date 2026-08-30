"""
DeFIGuard FastAPI Backend Service (Phase 2)
Integrates Hybrid XGBoost + GraphSAGE GNN + NTS Temporal Model with SHAP Explainability & Graph Generation.
"""
from pathlib import Path
import sys
import io
import json
import traceback
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Ensure Phase 1 scripts path is importable
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_PATH = (PROJECT_ROOT / "Phase 1" / "scripts").resolve()
SAMPLES_PATH = (PROJECT_ROOT / "data" / "samples").resolve()

if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

try:
    from inference import DefiGuardInference
except Exception as e:
    DefiGuardInference = None
    print(f"Warning: Could not import DefiGuardInference: {e}")

app = FastAPI(
    title="DeFIGuard AML Investigation API",
    description="Explainable Anti-Money Laundering Analytics for DeFi Transactions",
    version="2.0.0",
)

# Allow CORS for all local dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_engine: Optional[DefiGuardInference] = None


@app.on_event("startup")
def startup_event():
    global model_engine
    if DefiGuardInference is None:
        print("ERROR: DefiGuardInference class not available.")
        return
    try:
        print("Loading DeFIGuard ML Inference Engine...")
        model_engine = DefiGuardInference()
        print("DeFIGuard ML Engine loaded successfully!")
    except Exception as e:
        print(f"Error loading inference engine: {e}")
        traceback.print_exc()


def _sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean NaN and inf values so they are JSON compliant."""
    cleaned = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = 0.0
            elif isinstance(v, (np.integer, np.int64, np.int32)):
                row[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                row[k] = float(v)
            else:
                row[k] = v
        cleaned.append(row)
    return cleaned


def _compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total_txns = len(df)
    if total_txns == 0:
        return {}

    flagged_df = df[df["aml_flag"] == 1]
    flagged_count = len(flagged_df)
    alert_rate = (flagged_count / total_txns) * 100.0 if total_txns > 0 else 0.0

    total_volume = float(pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0).sum())
    flagged_volume = float(pd.to_numeric(flagged_df.get("amount", 0.0), errors="coerce").fillna(0.0).sum())

    # Pattern breakdown
    pattern_counts = df["pattern_type"].value_counts().to_dict()

    # Risk score distribution histogram (5 brackets)
    scores = pd.to_numeric(df["suspicion_score"], errors="coerce").fillna(0.0)
    brackets = {
        "0.0 - 0.2": int((scores < 0.2).sum()),
        "0.2 - 0.4": int(((scores >= 0.2) & (scores < 0.4)).sum()),
        "0.4 - 0.6": int(((scores >= 0.4) & (scores < 0.6)).sum()),
        "0.6 - 0.8": int(((scores >= 0.6) & (scores < 0.8)).sum()),
        "0.8 - 1.0": int((scores >= 0.8).sum()),
    }

    # Top high risk wallets
    flagged_wallets = set(flagged_df.get("from", pd.Series(dtype=str)).dropna()).union(
        set(flagged_df.get("to", pd.Series(dtype=str)).dropna())
    )

    return {
        "total_transactions": total_txns,
        "flagged_transactions": flagged_count,
        "clean_transactions": total_txns - flagged_count,
        "alert_rate_percentage": round(alert_rate, 2),
        "total_volume_eth": round(total_volume, 4),
        "flagged_volume_eth": round(flagged_volume, 4),
        "high_risk_wallets_count": len(flagged_wallets),
        "average_suspicion_score": round(float(scores.mean()), 4),
        "pattern_distribution": pattern_counts,
        "score_distribution": brackets,
    }


@app.get("/")
def root():
    return {
        "service": "DeFIGuard AML Investigation API",
        "status": "online",
        "version": "2.0.0",
        "endpoints": {
            "predict_csv": "POST /predict",
            "predict_json": "POST /predict-json",
            "sample_datasets": "GET /samples",
            "get_sample": "GET /samples/{name}",
            "model_info": "GET /model-info",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_engine is not None,
        "decision_threshold": model_engine.threshold if model_engine else None,
        "features_count": len(model_engine.feature_names) if model_engine else 0,
    }


@app.get("/model-info")
def model_info():
    """Returns Phase 1 validation benchmarks, architecture breakdown, and threshold info."""
    reports_file = PROJECT_ROOT / "Phase 1" / "Phase 1 Reports" / "test_metrics.json"
    metrics = {}
    if reports_file.exists():
        with open(reports_file, "r") as f:
            metrics = json.load(f)

    return {
        "model_type": "Hybrid Fusion (XGBoost + GraphSAGE GNN + NTS Temporal Intelligence)",
        "gold_standard_metrics": {
            "roc_auc": metrics.get("test_roc_auc", 0.933),
            "recall": metrics.get("test_recall", 0.9686),
            "precision": metrics.get("test_precision", 0.4427),
            "f1_score": metrics.get("test_f1", 0.6076),
            "pr_auc": metrics.get("test_pr_auc", 0.4694),
            "decision_threshold": metrics.get("decision_threshold", 0.4485),
        },
        "topology_coverage": [
            {
                "pattern": "Smurfing",
                "topology": "Fan-out / Fan-in (1 -> N -> 1)",
                "primary_detector": "GraphSAGE GNN Structural Embeddings",
            },
            {
                "pattern": "Peel Chain",
                "topology": "Linear Forwarding (A -> B -> C -> D...)",
                "primary_detector": "NTS Temporal Intelligence + XGBoost",
            },
            {
                "pattern": "Circular Ring",
                "topology": "Cyclic (A -> B -> C -> A)",
                "primary_detector": "NTS Burstiness + Baseline Degree",
            },
        ],
        "feature_count": len(model_engine.feature_names) if model_engine else 51,
        "features": model_engine.feature_names if model_engine else [],
    }


@app.get("/samples")
def list_samples():
    """Returns available pre-packaged sample transaction datasets for rapid testing."""
    samples = [
        {
            "id": "sample_100k_test",
            "title": "100k Gold Standard Test Sample (Peel + Smurf + Circular + Normal)",
            "description": "Representative slice of 120 transactions from the untouched 100k test set with full feature set.",
            "row_count": 120,
            "filename": "sample_100k_test.csv",
        },
        {
            "id": "sample_raw_transfers",
            "title": "Raw Ethereum Transfers (tx_hash, from, to, amount, block_time)",
            "description": "Un-engineered raw transfers testing real-time feature computation and GNN wallet lookup.",
            "row_count": 120,
            "filename": "sample_raw_transfers.csv",
        },
    ]
    return {"samples": samples}


@app.get("/samples/{sample_id}")
def get_sample_data(sample_id: str):
    """Loads and predicts directly on a preloaded sample dataset."""
    if model_engine is None:
        raise HTTPException(status_code=500, detail="ML model engine not loaded")

    csv_path = SAMPLES_PATH / f"{sample_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found")

    df = pd.read_csv(csv_path)
    results_df = model_engine.predict(df)
    graph_payload = model_engine.build_network_graph(results_df)
    summary_stats = _compute_summary_stats(results_df)

    records = _sanitize_records(results_df.to_dict(orient="records"))

    return {
        "sample_id": sample_id,
        "count": len(records),
        "summary": summary_stats,
        "graph": graph_payload,
        "results": records,
    }


@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    """Accepts a CSV file of transactions, executes ML inference, computes SHAP explanations and generates graph network."""
    if model_engine is None:
        raise HTTPException(status_code=500, detail="ML Model Engine is not loaded.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")

    if len(df) == 0:
        return {"count": 0, "summary": {}, "graph": {"nodes": [], "edges": []}, "results": []}

    try:
        results_df = model_engine.predict(df)
        graph_payload = model_engine.build_network_graph(results_df)
        summary_stats = _compute_summary_stats(results_df)
        records = _sanitize_records(results_df.to_dict(orient="records"))

        return {
            "filename": file.filename,
            "count": len(records),
            "summary": summary_stats,
            "graph": graph_payload,
            "results": records,
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Inference execution error: {e}\n{tb}")


class TransactionBatch(BaseModel):
    transactions: List[Dict[str, Any]]


@app.post("/predict-json")
def predict_json(payload: TransactionBatch):
    """Predicts on a JSON batch of transactions."""
    if model_engine is None:
        raise HTTPException(status_code=500, detail="ML Model Engine is not loaded.")

    if not payload.transactions:
        return {"count": 0, "summary": {}, "graph": {"nodes": [], "edges": []}, "results": []}

    try:
        df = pd.DataFrame(payload.transactions)
        results_df = model_engine.predict(df)
        graph_payload = model_engine.build_network_graph(results_df)
        summary_stats = _compute_summary_stats(results_df)
        records = _sanitize_records(results_df.to_dict(orient="records"))

        return {
            "count": len(records),
            "summary": summary_stats,
            "graph": graph_payload,
            "results": records,
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Inference execution error: {e}\n{tb}")
