"""
DeFIGuard Universal Ethereum Dataset Normalizer
Standardizes heterogeneous Ethereum transaction formats (Etherscan, Dune, BigQuery, Kaggle, raw node RPC)
into the exact schema and feature space expected by the DeFIGuard ML inference engine.
"""

from typing import Dict, Any, Tuple, Optional, List
import re
import numpy as np
import pandas as pd
import networkx as nx


class EthereumDatasetNormalizer:
    """
    Normalizes any Ethereum transaction dataset into DeFIGuard's canonical format.
    
    Handles:
    1. Schema auto-detection & column aliasing (Etherscan, Dune, BigQuery, etc.)
    2. Unit conversion (Wei <-> ETH automatic scaling)
    3. Address canonicalization (checksum lowercase, contract creation handling)
    4. Temporal parsing (Unix epoch seconds/millis, ISO 8601, UTC strings)
    5. On-the-fly graph topological feature engineering (degrees, flow ratio, PageRank, betweenness)
    """

    # Recognized column aliases (all lowercase for case-insensitive matching)
    COLUMN_ALIASES = {
        "tx_hash": [
            "tx_hash", "hash", "txhash", "transaction_hash", "tx_id", "txid",
            "transaction_id", "transaction hash", "txn_hash"
        ],
        "from": [
            "from", "from_address", "origin_from_address", "source", "sender", "src", "from_wallet",
            "from address", "origin"
        ],
        "to": [
            "to", "to_address", "origin_to_address", "target", "recipient", "receiver", "dst",
            "to_wallet", "to address", "destination"
        ],
        "amount": [
            "amount", "value", "value_in(eth)", "value_out(eth)", "eth_value",
            "val", "quantity", "value (eth)", "amount_eth", "amount_precise_raw", "amount_raw_precise"
        ],
        "block_number": [
            "block_number", "blockno", "block_num", "block", "blocknumber",
            "block height", "block_height"
        ],
        "block_time": [
            "block_time", "block_timestamp", "timestamp", "datetime (utc)",
            "datetime", "unixtimestamp", "time", "date", "tx_time"
        ]
    }

    REQUIRED_GRAPH_FEATURES = [
        "from_in_degree", "from_out_degree", "from_flow_ratio", "from_pagerank", "from_betweenness",
        "to_in_degree", "to_out_degree", "to_flow_ratio", "to_pagerank", "to_betweenness"
    ]

    def __init__(self):
        pass

    def detect_schema_source(self, columns: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Infers the source format of the dataset based on column naming signatures
        and returns the source name along with a column mapping dictionary.
        """
        raw_cols_lower = {str(c).strip().lower(): c for c in columns}
        mapping = {}

        has_etherscan_sig = any("value_in(eth)" in c or "datetime (utc)" in c or "txnhash" in c or "txfee(eth)" in c for c in raw_cols_lower)
        has_bigquery_sig = "hash" in raw_cols_lower and "from_address" in raw_cols_lower and "to_address" in raw_cols_lower and "value" in raw_cols_lower
        has_flipside_sig = ("from_address" in raw_cols_lower or "origin_from_address" in raw_cols_lower) and ("amount" in raw_cols_lower or "amount_usd" in raw_cols_lower)
        has_dune_sig = "tx_hash" in raw_cols_lower and "block_time" in raw_cols_lower and "value" in raw_cols_lower
        has_native_sig = all(k in raw_cols_lower for k in ["tx_hash", "from", "to", "amount"])

        if has_etherscan_sig:
            detected_source = "Etherscan Export CSV"
        elif has_bigquery_sig:
            detected_source = "Google BigQuery (crypto_ethereum)"
        elif has_flipside_sig:
            detected_source = "Flipside Crypto Export"
        elif has_dune_sig:
            detected_source = "Dune Analytics Export"
        elif has_native_sig:
            detected_source = "DeFIGuard Native / Processed"
        else:
            detected_source = "Custom / Generic Ethereum CSV"

        # 2. Build column mapping via alias dictionary
        # Special priority: for Etherscan with both Value_IN(ETH) and Value_OUT(ETH)
        etherscan_val_in = raw_cols_lower.get("value_in(eth)")
        etherscan_val_out = raw_cols_lower.get("value_out(eth)")

        for canonical, aliases in self.COLUMN_ALIASES.items():
            if canonical == "amount" and (etherscan_val_in or etherscan_val_out):
                # Etherscan special handling
                mapping["amount"] = etherscan_val_in if etherscan_val_in else etherscan_val_out
                continue

            for alias in aliases:
                # Direct match
                if alias in raw_cols_lower:
                    mapping[canonical] = raw_cols_lower[alias]
                    break
                # Relaxed match (removing underscores, spaces, dashes)
                norm_alias = re.sub(r"[_\s\-]+", "", alias)
                matched_col = None
                for c_low, orig in raw_cols_lower.items():
                    if re.sub(r"[_\s\-]+", "", c_low) == norm_alias:
                        matched_col = orig
                        break
                if matched_col:
                    mapping[canonical] = matched_col
                    break

        return detected_source, mapping

    def _clean_addresses(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Normalizes addresses to lowercase and handles contract creations or empty addresses.
        """
        cleaned_df = df.copy()
        
        # Convert to string and lowercase
        cleaned_df["from"] = cleaned_df["from"].astype(str).str.strip().str.lower()
        cleaned_df["to"] = cleaned_df["to"].astype(str).str.strip().str.lower()

        # Handle null/nan/none string representations
        invalid_mask = cleaned_df["from"].isin(["nan", "none", "", "0x", "null"])
        dropped_count = int(invalid_mask.sum())
        if dropped_count > 0:
            cleaned_df.loc[invalid_mask, "from"] = "0x0000000000000000000000000000000000000000"

        # In Ethereum, null 'to' address signifies contract creation
        contract_creation_mask = cleaned_df["to"].isin(["nan", "none", "", "0x", "null"])
        cleaned_df.loc[contract_creation_mask, "to"] = "0x0000000000000000000000000000000000000000"

        return cleaned_df, dropped_count

    def _normalize_amounts(self, df: pd.DataFrame) -> Tuple[pd.Series, str]:
        """
        Detects whether amounts are in Wei, Gwei, or ETH, and standardizes to float ETH.
        """
        raw_amount = df["amount"]

        # If values are hexadecimal strings (e.g., '0x1bc16d674ec80000')
        def parse_val(v):
            if pd.isna(v):
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            v_str = str(v).strip()
            if v_str.startswith("0x") or v_str.startswith("0X"):
                try:
                    return float(int(v_str, 16))
                except Exception:
                    return 0.0
            try:
                return float(v_str.replace(",", ""))
            except Exception:
                return 0.0

        numeric_vals = raw_amount.map(parse_val).fillna(0.0)
        non_zero = numeric_vals[numeric_vals > 0]

        if len(non_zero) == 0:
            return numeric_vals, "Zero / Empty values"

        # Check if values are in Wei (typical Wei transfers: > 10^12)
        # Even 0.0001 ETH = 10^14 Wei. If median of non-zero values > 10^10, it's Wei.
        median_val = float(non_zero.median())
        max_val = float(non_zero.max())

        if median_val >= 1e10 or max_val >= 1e14:
            standardized_eth = numeric_vals / 1e18
            unit_action = "Auto-converted Wei -> ETH (divided by 10^18)"
        elif median_val >= 1e4:
            # Possible Gwei
            standardized_eth = numeric_vals / 1e9
            unit_action = "Auto-converted Gwei -> ETH (divided by 10^9)"
        else:
            standardized_eth = numeric_vals
            unit_action = "Preserved native ETH scale"

        return standardized_eth.astype(float), unit_action

    def _normalize_timestamps(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Parses timestamps from strings or Unix epochs into datetime, hour_of_day, and day_of_week.
        """
        raw_time = df.get("block_time")

        if raw_time is None or raw_time.isna().all():
            # If no time provided, default to dummy sequence
            now = pd.Timestamp.utcnow()
            times = pd.Series([now] * len(df), index=df.index)
        else:
            sample_val = raw_time.dropna().iloc[0] if len(raw_time.dropna()) > 0 else None
            is_numeric = False
            if sample_val is not None:
                try:
                    float(sample_val)
                    is_numeric = True
                except (ValueError, TypeError):
                    is_numeric = False

            if is_numeric:
                numeric_times = pd.to_numeric(raw_time, errors="coerce").fillna(0)
                # Check seconds vs milliseconds (millis > 1e11)
                if numeric_times.max() > 1e11:
                    times = pd.to_datetime(numeric_times, unit="ms", utc=True, errors="coerce")
                else:
                    times = pd.to_datetime(numeric_times, unit="s", utc=True, errors="coerce")
            else:
                times = pd.to_datetime(raw_time, utc=True, errors="coerce")

        times = times.fillna(pd.Timestamp.utcnow())
        hour_of_day = times.dt.hour.astype(float)
        day_of_week = times.dt.dayofweek.astype(float)
        iso_times = times.dt.strftime("%Y-%m-%d %H:%M:%S.000 UTC")

        return iso_times, hour_of_day, day_of_week

    def _compute_graph_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Dynamically computes topological graph metrics on the transaction batch using NetworkX.
        Calculates: in_degree, out_degree, flow_ratio, PageRank, betweenness centrality.
        """
        graph = nx.DiGraph()
        # Add all transaction edges
        edges = list(zip(df["from"].astype(str), df["to"].astype(str)))
        graph.add_edges_from(edges)

        unique_wallets_count = graph.number_of_nodes()

        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())

        # PageRank (alpha=0.85, max_iter=100)
        try:
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except Exception:
            pagerank = {w: 1.0 / max(1, unique_wallets_count) for w in graph.nodes()}

        # Betweenness centrality (approximate with k=min(100, N) for fast responsiveness)
        k_sample = min(100, unique_wallets_count)
        try:
            betweenness = nx.betweenness_centrality(graph, k=k_sample, seed=42)
        except Exception:
            betweenness = {w: 0.0 for w in graph.nodes()}

        wallets = list(graph.nodes())
        wf = pd.DataFrame({"wallet": wallets})
        wf["in_degree"] = wf["wallet"].map(in_degree).fillna(0).astype(int)
        wf["out_degree"] = wf["wallet"].map(out_degree).fillna(0).astype(int)

        tot_deg = wf["in_degree"] + wf["out_degree"]
        wf["flow_ratio"] = wf["out_degree"].where(tot_deg > 0, 0.0) / tot_deg.where(tot_deg > 0, 1.0)
        wf["pagerank"] = wf["wallet"].map(pagerank).fillna(0.0).astype(float)
        wf["betweenness"] = wf["wallet"].map(betweenness).fillna(0.0).astype(float)

        wf_dict = wf.set_index("wallet").to_dict(orient="index")

        # Map back to transactions
        from_feats = [wf_dict.get(w, {"in_degree": 0, "out_degree": 0, "flow_ratio": 0.0, "pagerank": 0.0, "betweenness": 0.0}) for w in df["from"]]
        to_feats = [wf_dict.get(w, {"in_degree": 0, "out_degree": 0, "flow_ratio": 0.0, "pagerank": 0.0, "betweenness": 0.0}) for w in df["to"]]

        from_df = pd.DataFrame(from_feats, index=df.index).add_prefix("from_")
        to_df = pd.DataFrame(to_feats, index=df.index).add_prefix("to_")

        graph_features_df = pd.concat([from_df, to_df], axis=1)
        return graph_features_df, unique_wallets_count

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main entry point: ingests an arbitrary DataFrame, normalizes columns,
        standardizes units, temporal features, and generates missing graph features.
        
        Returns:
            normalized_df: DataFrame ready for DeFIGuard ML inference.
            metadata: Diagnostic information on schema detection and transformations.
        """
        if len(df) == 0:
            return df.copy(), {
                "detected_source": "Empty Dataset",
                "columns_mapped": {},
                "unit_conversion": "N/A",
                "graph_features_generated": False,
                "wallet_count": 0,
                "transaction_count": 0,
            }

        # 1. Detect schema source and column mapping
        detected_source, col_map = self.detect_schema_source(list(df.columns))

        # Check required columns
        if "from" not in col_map or "to" not in col_map:
            raise ValueError(
                f"Dataset normalization failed: Could not identify sender ('from') or recipient ('to') columns. "
                f"Available columns: {list(df.columns)}"
            )

        norm_df = pd.DataFrame(index=df.index)

        # 2. Map standard transaction columns
        # Transaction Hash
        if "tx_hash" in col_map:
            norm_df["tx_hash"] = df[col_map["tx_hash"]].astype(str)
        else:
            norm_df["tx_hash"] = [f"tx_{i:06d}" for i in range(len(df))]

        # From / To Wallets
        norm_df["from"] = df[col_map["from"]]
        norm_df["to"] = df[col_map["to"]]
        norm_df, dropped_rows = self._clean_addresses(norm_df)

        # Amount & Unit Scaling
        if "amount" in col_map:
            norm_df["amount"] = df[col_map["amount"]]
        else:
            norm_df["amount"] = 0.0

        standardized_amount, unit_action = self._normalize_amounts(norm_df)
        norm_df["amount"] = standardized_amount
        norm_df["amount_log1p"] = np.log1p(np.maximum(0.0, standardized_amount))

        # Block Number
        if "block_number" in col_map:
            norm_df["block_number"] = pd.to_numeric(df[col_map["block_number"]], errors="coerce").fillna(0).astype(int)
        else:
            norm_df["block_number"] = 0

        # Block Time & Temporal features
        if "block_time" in col_map:
            norm_df["block_time"] = df[col_map["block_time"]]
        else:
            norm_df["block_time"] = None

        iso_time, hour_of_day, day_of_week = self._normalize_timestamps(norm_df)
        norm_df["block_time"] = iso_time
        norm_df["hour_of_day"] = hour_of_day
        norm_df["day_of_week"] = day_of_week

        # 3. Check for existing graph features or generate them dynamically
        missing_graph_cols = [c for c in self.REQUIRED_GRAPH_FEATURES if c not in df.columns]
        graph_generated = False
        
        if len(missing_graph_cols) > 0:
            # Generate graph features on-the-fly
            graph_feats_df, wallet_count = self._compute_graph_features(norm_df)
            for c in self.REQUIRED_GRAPH_FEATURES:
                norm_df[c] = graph_feats_df[c]
            graph_generated = True
        else:
            # Preserve existing graph features from processed datasets
            for c in self.REQUIRED_GRAPH_FEATURES:
                norm_df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            wallet_count = len(set(norm_df["from"]).union(set(norm_df["to"])))

        # Preserve any existing labels or pattern tags if present (for evaluation/benchmarks)
        for passthrough_col in ["aml_label", "label_note", "pattern", "pattern_type", "split"]:
            if passthrough_col in df.columns:
                norm_df[passthrough_col] = df[passthrough_col]

        metadata = {
            "detected_source": detected_source,
            "columns_mapped": col_map,
            "unit_conversion": unit_action,
            "graph_features_generated": graph_generated,
            "wallet_count": wallet_count,
            "transaction_count": len(norm_df),
            "cleaned_addresses_count": dropped_rows,
        }

        return norm_df, metadata
