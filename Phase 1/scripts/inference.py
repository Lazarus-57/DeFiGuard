import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = SCRIPT_DIR.parent / "Phase 1 Models"


class DefiGuardInference:
    def __init__(self, models_dir: str | Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        
        # Load XGBoost Model
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.models_dir / "master_hybrid_model.json")
        
        # Load Feature Names
        with open(self.models_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)
            
        # Load Threshold
        with open(self.models_dir / "decision_threshold.json", "r") as f:
            self.threshold = json.load(f)["threshold"]
            
        # Load NTS Map
        with open(self.models_dir / "nts_map.pkl", "rb") as f:
            self.nts_map = pickle.load(f)
            
        # Load GNN Embeddings
        with open(self.models_dir / "gnn_wallet_embeddings.pkl", "rb") as f:
            gnn_data = pickle.load(f)
            self.wallet_to_idx = gnn_data["wallet_to_idx"]
            self.gnn_embeddings = gnn_data["embeddings"]
            
        # Load SHAP Explainer (with resilient fallback to model-based TreeExplainer)
        try:
            with open(self.models_dir / "shap_explainer.pkl", "rb") as f:
                self.explainer = pickle.load(f)
        except Exception:
            self.explainer = shap.TreeExplainer(self.model)

    def _prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = {"transaction_id", "tx_hash", "aml_label", "label_note", "split", "block_time", "from", "to", "pattern", "pattern_id"}
        feature_cols = [c for c in df.columns if c not in drop_cols and c in self.feature_names]
        X = df[feature_cols].copy()
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            
        # If amount_log1p or time features are missing in raw uploads, compute them
        if "amount_log1p" not in X.columns and "amount" in df.columns:
            X["amount_log1p"] = np.log1p(pd.to_numeric(df["amount"], errors="coerce").fillna(0.0))
            
        if "hour_of_day" not in X.columns and "block_time" in df.columns:
            try:
                times = pd.to_datetime(df["block_time"], errors="coerce")
                X["hour_of_day"] = times.dt.hour.fillna(0).astype(float)
                X["day_of_week"] = times.dt.dayofweek.fillna(0).astype(float)
            except Exception:
                X["hour_of_day"] = 0.0
                X["day_of_week"] = 0.0
            
        # Ensure all required base features are present
        for col in self.feature_names:
            if not col.startswith("gnn_") and not col.startswith("nts_") and not col.endswith("_nts") and col not in X.columns:
                X[col] = 0.0
                
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _attach_gnn_embeddings(self, base_X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        emb_dim = self.gnn_embeddings.shape[1]

        def get_emb(wallet):
            idx = self.wallet_to_idx.get(wallet)
            if idx is None:
                return np.zeros(emb_dim, dtype=np.float32)
            return self.gnn_embeddings[idx]

        from_matrix = np.vstack([get_emb(w) for w in df["from"].fillna("").tolist()]).astype(np.float32)
        to_matrix = np.vstack([get_emb(w) for w in df["to"].fillna("").tolist()]).astype(np.float32)

        gnn_cols = {}
        for i in range(emb_dim):
            gnn_cols[f"gnn_from_emb_{i}"] = from_matrix[:, i]
        for i in range(emb_dim):
            gnn_cols[f"gnn_to_emb_{i}"] = to_matrix[:, i]

        gnn_df = pd.DataFrame(gnn_cols, index=df.index)
        return pd.concat([base_X, gnn_df], axis=1)

    def _attach_nts_features(self, base_X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        from_wallet = df["from"].fillna("")
        to_wallet = df["to"].fillna("")

        from_nts = from_wallet.map(self.nts_map).fillna(0.0).astype(float)
        to_nts = to_wallet.map(self.nts_map).fillna(0.0).astype(float)

        nts_df = pd.DataFrame(
            {
                "from_nts": from_nts,
                "to_nts": to_nts,
                "nts_max": np.maximum(from_nts, to_nts),
                "nts_mean": (from_nts + to_nts) / 2.0,
            },
            index=df.index,
        )
        return pd.concat([base_X, nts_df], axis=1)

    def _generate_narrative_explanation(self, top_reasons: list, pattern_type: str, gnn_score: float, nts_score: float, base_score: float, score: float) -> str:
        """Generates investigator-friendly narrative explaining why a transaction was flagged."""
        if score < self.threshold:
            return "Transaction exhibits standard baseline behavior within normal network thresholds."

        reasons_text = ", ".join(f"{r['feature']} (+{r['importance']:.3f})" for r in top_reasons[:3])
        
        if "Smurfing" in pattern_type:
            return (
                f"[HIGH RISK] Identified as Smurfing topology. Wallet graph neighborhood exhibits structural dispersion/fan-out "
                f"characteristics captured by GNN node embeddings. Key risk drivers: {reasons_text}."
            )
        elif "Peel/Circular" in pattern_type:
            return (
                f"[HIGH RISK] Identified as Peel Chain / Circular Ring. Demonstrates extreme temporal burstiness & rapid forwarding "
                f"captured by NTS metrics. Key risk drivers: {reasons_text}."
            )
        else:
            return (
                f"[HIGH RISK] Flagged by Baseline ML features (unusual volume, centrality, or flow imbalance). "
                f"Key risk drivers: {reasons_text}."
            )

    def predict(self, df: pd.DataFrame, compute_detailed_shap: bool = True) -> pd.DataFrame:
        """
        Runs inference on a DataFrame of transactions.
        
        Input: DataFrame with columns ['tx_hash', 'from', 'to', 'amount', ...]
        Output: Same DataFrame with added columns:
            - suspicion_score (float 0-1)
            - aml_flag (int 0/1)
            - pattern_type (str: 'Normal', 'Smurfing (Structural)', 'Peel/Circular (Temporal)', 'Generic Baseline')
            - top_shap_reason (str)
            - shap_breakdown (dict: top positive/negative drivers, component sums, narrative)
        """
        if len(df) == 0:
            return df.copy()
            
        results_df = df.copy()
        
        # 1. Feature Engineering
        X_base = self._prepare_base_features(results_df)
        X_gnn = self._attach_gnn_embeddings(X_base, results_df)
        X_final = self._attach_nts_features(X_gnn, results_df)
        
        # Ensure column order matches training EXACTLY
        X_final = X_final[self.feature_names]

        # 2. Model Prediction
        y_prob = self.model.predict_proba(X_final)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        
        results_df["suspicion_score"] = np.round(y_prob, 5)
        results_df["aml_flag"] = y_pred.astype(int)
        
        # 3. SHAP Explainability & Pattern Classification
        flagged_idx = np.where(y_pred == 1)[0]
        
        pattern_types = ["Normal"] * len(results_df)
        top_reasons = ["Normal Baseline"] * len(results_df)
        shap_breakdowns = [None] * len(results_df)
        
        if len(flagged_idx) > 0 and compute_detailed_shap:
            X_flagged = X_final.iloc[flagged_idx]
            shap_raw = self.explainer.shap_values(X_flagged)
            
            # Handle different shap output shapes (binary classification)
            if isinstance(shap_raw, list):
                shap_matrix = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
            elif hasattr(shap_raw, "values"):
                shap_matrix = shap_raw.values
            else:
                shap_matrix = np.array(shap_raw)
                
            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, 1]
            
            for i, local_idx in enumerate(flagged_idx):
                shaps = shap_matrix[i]
                
                # Decompose into 3 layers
                gnn_sum = float(sum(shaps[j] for j, feat in enumerate(self.feature_names) if feat.startswith("gnn_")))
                nts_sum = float(sum(shaps[j] for j, feat in enumerate(self.feature_names) if feat.startswith("nts_") or feat.endswith("_nts")))
                base_sum = float(sum(shaps[j] for j, feat in enumerate(self.feature_names) if not (feat.startswith("gnn_") or feat.startswith("nts_") or feat.endswith("_nts"))))
                
                # Classify pattern
                if gnn_sum > nts_sum and gnn_sum > 0:
                    pat = "Smurfing (Structural)"
                elif nts_sum > gnn_sum and nts_sum > 0:
                    pat = "Peel/Circular (Temporal)"
                else:
                    pat = "Generic Baseline"
                    
                pattern_types[local_idx] = pat
                
                # Rank top features
                sorted_feat_indices = np.argsort(shaps)[::-1]
                top_features = []
                for idx in sorted_feat_indices[:6]:
                    feat_val = float(X_final.iloc[local_idx][self.feature_names[idx]])
                    top_features.append({
                        "feature": self.feature_names[idx],
                        "importance": float(np.round(shaps[idx], 4)),
                        "feature_value": float(np.round(feat_val, 4)) if not np.isnan(feat_val) else 0.0
                    })
                    
                top_reasons[local_idx] = top_features[0]["feature"] if top_features else "N/A"
                
                narrative = self._generate_narrative_explanation(
                    top_features, pat, gnn_sum, nts_sum, base_sum, float(y_prob[local_idx])
                )
                
                shap_breakdowns[local_idx] = {
                    "top_drivers": top_features,
                    "gnn_structural_score": round(gnn_sum, 4),
                    "nts_temporal_score": round(nts_sum, 4),
                    "base_features_score": round(base_sum, 4),
                    "dominant_layer": "GNN Structural" if gnn_sum >= max(nts_sum, base_sum) else ("NTS Temporal" if nts_sum >= base_sum else "Base Centrality/Amount"),
                    "narrative": narrative
                }
                
        # Fill default for non-flagged transactions
        for i in range(len(results_df)):
            if shap_breakdowns[i] is None:
                shap_breakdowns[i] = {
                    "top_drivers": [],
                    "gnn_structural_score": 0.0,
                    "nts_temporal_score": 0.0,
                    "base_features_score": 0.0,
                    "dominant_layer": "Normal Baseline",
                    "narrative": "Transaction falls below AML threshold. Normal behavior detected."
                }
                
        results_df["pattern_type"] = pattern_types
        results_df["top_shap_reason"] = top_reasons
        results_df["shap_breakdown"] = shap_breakdowns
        
        return results_df

    def build_network_graph(self, results_df: pd.DataFrame, max_nodes: int = 150) -> dict:
        """
        Builds a node-link network graph for visualization in the frontend.
        """
        if len(results_df) == 0:
            return {"nodes": [], "edges": [], "stats": {}}

        # Prioritize flagged transactions first, then fill with other transactions
        flagged_df = results_df[results_df["aml_flag"] == 1]
        clean_df = results_df[results_df["aml_flag"] == 0]
        
        sample_df = pd.concat([flagged_df, clean_df.head(max(0, max_nodes - len(flagged_df)))]).head(max_nodes)
        
        wallet_stats = {}
        edges = []

        for _, row in sample_df.iterrows():
            src = str(row.get("from", "unknown"))
            dst = str(row.get("to", "unknown"))
            amount = float(row.get("amount", 0.0))
            is_flagged = bool(row.get("aml_flag", 0) == 1)
            score = float(row.get("suspicion_score", 0.0))
            pattern = str(row.get("pattern_type", "Normal"))
            tx_hash = str(row.get("tx_hash", ""))
            
            # Init wallet stats
            for w in (src, dst):
                if w not in wallet_stats:
                    wallet_stats[w] = {
                        "id": w,
                        "address": w,
                        "label": f"{w[:6]}...{w[-4:]}" if len(w) > 10 else w,
                        "inDegree": 0,
                        "outDegree": 0,
                        "totalVolume": 0.0,
                        "suspiciousCount": 0,
                        "maxScore": 0.0,
                        "isSuspicious": False,
                    }
                    
            wallet_stats[src]["outDegree"] += 1
            wallet_stats[src]["totalVolume"] += amount
            wallet_stats[dst]["inDegree"] += 1
            wallet_stats[dst]["totalVolume"] += amount
            
            if is_flagged:
                wallet_stats[src]["suspiciousCount"] += 1
                wallet_stats[dst]["suspiciousCount"] += 1
                wallet_stats[src]["isSuspicious"] = True
                wallet_stats[dst]["isSuspicious"] = True
                
            wallet_stats[src]["maxScore"] = max(wallet_stats[src]["maxScore"], score)
            wallet_stats[dst]["maxScore"] = max(wallet_stats[dst]["maxScore"], score)

            edges.append({
                "id": tx_hash if tx_hash else f"{src}->{dst}_{len(edges)}",
                "source": src,
                "target": dst,
                "amount": round(amount, 4),
                "suspicion_score": round(score, 4),
                "is_flagged": is_flagged,
                "pattern_type": pattern,
            })

        nodes = list(wallet_stats.values())
        for n in nodes:
            n["totalVolume"] = round(n["totalVolume"], 4)
            n["maxScore"] = round(n["maxScore"], 4)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "suspicious_nodes": sum(1 for n in nodes if n["isSuspicious"]),
                "suspicious_edges": sum(1 for e in edges if e["is_flagged"]),
            }
        }

if __name__ == "__main__":
    print("Initializing DefiGuardInference module...")
    try:
        infer = DefiGuardInference()
        print("Successfully loaded all artifacts!")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

