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
            
        # Load SHAP Explainer
        with open(self.models_dir / "shap_explainer.pkl", "rb") as f:
            self.explainer = pickle.load(f)

    def _prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = {"transaction_id", "tx_hash", "aml_label", "label_note", "split", "block_time", "from", "to", "pattern", "pattern_id"}
        feature_cols = [c for c in df.columns if c not in drop_cols and c in self.feature_names]
        X = df[feature_cols].copy()
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            
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

        from_matrix = np.vstack([get_emb(w) for w in df["from"].tolist()]).astype(np.float32)
        to_matrix = np.vstack([get_emb(w) for w in df["to"].tolist()]).astype(np.float32)

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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on a DataFrame of transactions.
        
        Input: DataFrame with columns ['tx_hash', 'from', 'to', 'amount', ...]
        Output: Same DataFrame with added columns:
            - suspicion_score (float 0-1)
            - aml_flag (int 0/1)
            - pattern_type (str: 'Normal', 'Smurfing', 'Peel/Circular')
            - top_reason (str: the highest SHAP contribution)
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
        
        results_df["suspicion_score"] = y_prob
        results_df["aml_flag"] = y_pred
        
        # 3. SHAP Explainability & Pattern Classification
        # We only compute SHAP for flagged transactions to save time
        flagged_idx = np.where(y_pred == 1)[0]
        
        pattern_types = ["Normal"] * len(results_df)
        top_reasons = [""] * len(results_df)
        
        if len(flagged_idx) > 0:
            X_flagged = X_final.iloc[flagged_idx]
            shap_values = self.explainer.shap_values(X_flagged)
            
            for i, local_idx in enumerate(flagged_idx):
                shaps = shap_values[i]
                
                # Find the feature with the highest positive contribution
                top_feature_idx = np.argmax(shaps)
                top_feature_name = self.feature_names[top_feature_idx]
                top_reasons[local_idx] = top_feature_name
                
                # Classify pattern based on SHAP contributions
                gnn_sum = sum(shaps[j] for j, feat in enumerate(self.feature_names) if feat.startswith("gnn_"))
                nts_sum = sum(shaps[j] for j, feat in enumerate(self.feature_names) if feat.startswith("nts_") or feat.endswith("_nts"))
                
                if gnn_sum > nts_sum and gnn_sum > 0:
                    pattern_types[local_idx] = "Smurfing (Structural)"
                elif nts_sum > gnn_sum and nts_sum > 0:
                    pattern_types[local_idx] = "Peel/Circular (Temporal)"
                else:
                    pattern_types[local_idx] = "Generic Baseline"
                    
        results_df["pattern_type"] = pattern_types
        results_df["top_shap_reason"] = top_reasons
        
        return results_df

if __name__ == "__main__":
    # Simple self-test if run directly
    print("Initializing DefiGuardInference module...")
    try:
        infer = DefiGuardInference()
        print("Successfully loaded all artifacts!")
    except Exception as e:
        print(f"Error loading artifacts (ensure phase1_master_training.py has been run): {e}")
