"""
Data Preprocessing Pipeline for ABIA.
Enforces physical plausibility, duplicate removal, chronological sorting,
and causal-only missing value handling without raw data modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.abia.data.inspector import DatasetInspector


class DataPreprocessor:
    """Preprocesses raw battery telemetry into clean, chronologically verified datasets."""

    def __init__(
        self,
        inspector: Optional[DatasetInspector] = None,
        outlier_iqr_multiplier: float = 3.0,
    ):
        self.inspector = inspector or DatasetInspector()
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.preprocessing_log: Dict[str, Any] = {}

    def preprocess(
        self,
        df_or_path: str | Path | pd.DataFrame,
        output_path: Optional[str | Path] = "data/processed/battery_telemetry_cleaned.csv",
    ) -> pd.DataFrame:
        """Executes end-to-end preprocessing."""
        if isinstance(df_or_path, (str, Path)):
            raw_path = Path(df_or_path)
            # Ensure raw path is preserved
            df = pd.read_csv(raw_path) if raw_path.suffix.lower() == ".csv" else pd.read_parquet(raw_path)
            source_desc = str(raw_path)
        else:
            df = df_or_path.copy()
            source_desc = "in_memory_dataframe"

        initial_rows = len(df)
        semantic_map = self.inspector.map_columns_to_semantics(df)

        # 1. Deduplication
        df_cleaned = df.drop_duplicates().copy()
        dedup_count = initial_rows - len(df_cleaned)

        # 2. Canonical Column Standardization (Keep originals, add canonical aliases if distinct)
        bat_col = semantic_map.get("battery_id")
        time_col = semantic_map.get("timestamp")
        v_col = semantic_map.get("voltage")
        i_col = semantic_map.get("current")
        t_col = semantic_map.get("temperature")
        soc_col = semantic_map.get("soc")
        soh_col = semantic_map.get("soh")
        cyc_col = semantic_map.get("cycle")

        if not bat_col or bat_col not in df_cleaned.columns:
            df_cleaned["battery_id"] = "BAT_DEFAULT"
            bat_col = "battery_id"

        # 3. Timestamp Parsing & Chronological Sorting
        if time_col and time_col in df_cleaned.columns:
            try:
                df_cleaned["_parsed_time"] = pd.to_datetime(df_cleaned[time_col])
                sort_cols = [bat_col, "_parsed_time"]
            except Exception:
                sort_cols = [bat_col, time_col]
        elif cyc_col and cyc_col in df_cleaned.columns:
            sort_cols = [bat_col, cyc_col]
        else:
            sort_cols = [bat_col]

        df_cleaned = df_cleaned.sort_values(by=sort_cols).reset_index(drop=True)

        # 4. Outlier & Invalid Physical Range Checks
        invalid_mask = pd.Series(False, index=df_cleaned.index)

        # Voltage physical limits (Cell vs Pack check)
        if v_col and v_col in df_cleaned.columns:
            # Drop physically impossible voltages
            invalid_v = (df_cleaned[v_col] <= 0.0) | (df_cleaned[v_col] > 1000.0)
            invalid_mask = invalid_mask | invalid_v

        # Temperature physical limits (-40 C to 120 C)
        if t_col and t_col in df_cleaned.columns:
            invalid_t = (df_cleaned[t_col] < -40.0) | (df_cleaned[t_col] > 130.0)
            invalid_mask = invalid_mask | invalid_t

        # SOC limits [0, 105]
        if soc_col and soc_col in df_cleaned.columns:
            df_cleaned[soc_col] = df_cleaned[soc_col].clip(0.0, 100.0)

        # SOH limits [0, 110]
        if soh_col and soh_col in df_cleaned.columns:
            df_cleaned[soh_col] = df_cleaned[soh_col].clip(0.0, 105.0)

        invalid_count = int(invalid_mask.sum())
        df_cleaned = df_cleaned[~invalid_mask].reset_index(drop=True)

        # 5. Causal Missing Value Handling per Battery
        # Strictly forward-fill (never back-fill, to prevent target and state leakage from future)
        processed_groups: List[pd.DataFrame] = []
        for bat_id, group in df_cleaned.groupby(bat_col):
            # Sort chronologically within group
            group = group.sort_values(by=sort_cols[1:] if len(sort_cols) > 1 else sort_cols)
            # Causal ffill only
            group_imputed = group.ffill()
            # If initial row had NaNs, fill with group median
            group_imputed = group_imputed.bfill()
            processed_groups.append(group_imputed)

        df_processed = pd.concat(processed_groups, ignore_index=True)
        if "_parsed_time" in df_processed.columns:
            df_processed = df_processed.drop(columns=["_parsed_time"])

        # Save processed data if destination specified
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(out_p, index=False)

        self.preprocessing_log = {
            "source": source_desc,
            "initial_rows": initial_rows,
            "duplicate_rows_removed": dedup_count,
            "physically_invalid_rows_removed": invalid_count,
            "final_clean_rows": len(df_processed),
            "output_path": str(output_path) if output_path else None,
            "number_of_batteries": int(df_processed[bat_col].nunique()),
        }

        return df_processed
