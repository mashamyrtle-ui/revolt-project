"""
Automated Dataset Inspector for Battery Telemetry and Cycling Data.
Identifies physical and temporal semantics without assuming hardcoded column names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Canonical physical semantics and corresponding pattern matching keywords
CANONICAL_SEMANTICS: Dict[str, List[str]] = {
    "voltage": ["voltage", "volt", "v_cell", "vcell", "cell_voltage", "pack_v", "terminal_voltage", "v_meas", "v"],
    "current": ["current", "curr", "load_current", "amperes", "amp", "i_meas", "current_a", "i"],
    "temperature": ["temperature", "temp", "t_cell", "cell_temp", "temp_c", "temp_k", "thermocouple", "t"],
    "power": ["power", "p_meas", "watts", "power_w", "kw", "p"],
    "soc": ["soc", "state_of_charge", "stateofcharge", "soc_%", "soc_percent"],
    "soh": ["soh", "state_of_health", "stateofhealth", "soh_%", "health_pct", "soh_pct"],
    "cycle": ["cycle", "cycle_index", "cycle_number", "cycle_count", "cyc_cnt", "cyc"],
    "capacity": ["capacity", "cap", "ah", "capacity_ah", "discharge_capacity", "charge_capacity", "q_dis", "q_chg", "q"],
    "timestamp": ["time", "timestamp", "datetime", "date", "t_sec", "time_sec", "test_time", "relative_time", "step_time"],
    "battery_id": ["battery_id", "battery", "cell_id", "cell", "module_id", "module", "unit_id", "sample_id", "device_id"],
    "diagnostic": ["anomaly", "fault", "alarm", "warning", "status", "error", "label", "flag", "is_anomaly"],
}


class DatasetInspector:
    """Inspects arbitrary battery datasets and determines schema, data quality, and feature availability."""

    def __init__(self, semantic_map_override: Optional[Dict[str, str]] = None):
        self.semantic_map_override = semantic_map_override or {}

    def inspect_file_or_dataframe(
        self, data_source: str | Path | pd.DataFrame
    ) -> Dict[str, Any]:
        """Loads data if needed and performs full inspection."""
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            df = self._load_data(path)
            file_name = path.name
            file_size_bytes = path.stat().st_size
        else:
            df = data_source
            file_name = "in_memory_dataframe"
            file_size_bytes = df.memory_usage(deep=True).sum()

        return self.inspect_dataframe(df, file_name=file_name, file_size_bytes=int(file_size_bytes))

    def _load_data(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        elif suffix in [".json", ".jsonl"]:
            return pd.read_json(path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            # Fallback attempt CSV
            return pd.read_csv(path)

    def map_columns_to_semantics(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Maps canonical physical semantics to matching dataframe column names."""
        mapping: Dict[str, Optional[str]] = {}
        cols_lower = {col: str(col).strip().lower() for col in df.columns}

        # Apply overrides first
        for semantic, col_name in self.semantic_map_override.items():
            if col_name in df.columns:
                mapping[semantic] = col_name

        for semantic, keywords in CANONICAL_SEMANTICS.items():
            if semantic in mapping:
                continue

            matched_col: Optional[str] = None
            # Exact match first
            for orig_col, lcol in cols_lower.items():
                if lcol in keywords:
                    matched_col = orig_col
                    break

            # Substring match if exact match not found
            if not matched_col:
                for kw in keywords:
                    for orig_col, lcol in cols_lower.items():
                        # Avoid single-letter False positives like 'v' matching 'every'
                        if len(kw) == 1:
                            if lcol == kw or f"_{kw}" in lcol or f"{kw}_" in lcol:
                                matched_col = orig_col
                                break
                        elif kw in lcol:
                            matched_col = orig_col
                            break
                    if matched_col:
                        break

            mapping[semantic] = matched_col

        return mapping

    def inspect_dataframe(
        self, df: pd.DataFrame, file_name: str = "dataset", file_size_bytes: int = 0
    ) -> Dict[str, Any]:
        """Runs comprehensive inspection on a loaded dataframe."""
        semantic_map = self.map_columns_to_semantics(df)

        total_rows = len(df)
        total_cols = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())

        column_details: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = float((null_count / total_rows) * 100) if total_rows > 0 else 0.0

            stats: Dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_percentage": round(null_pct, 3),
            }

            if pd.api.types.is_numeric_dtype(series):
                valid_num = series.dropna()
                stats["min"] = float(valid_num.min()) if len(valid_num) > 0 else None
                stats["max"] = float(valid_num.max()) if len(valid_num) > 0 else None
                stats["mean"] = float(valid_num.mean()) if len(valid_num) > 0 else None
                stats["std"] = float(valid_num.std()) if len(valid_num) > 1 else None
                stats["infinite_count"] = int(np.isinf(valid_num).sum()) if len(valid_num) > 0 else 0
            else:
                stats["unique_values"] = int(series.nunique())

            column_details[str(col)] = stats

        # Battery / Module ID analysis
        battery_col = semantic_map.get("battery_id")
        if battery_col and battery_col in df.columns:
            battery_ids = [str(x) for x in df[battery_col].dropna().unique()]
            num_batteries = len(battery_ids)
        else:
            battery_ids = ["default_battery_01"]
            num_batteries = 1

        # Timestamp & Chronological analysis
        time_col = semantic_map.get("timestamp")
        time_available = bool(time_col and time_col in df.columns)
        chronological_coverage: Dict[str, Any] = {"available": time_available}
        estimated_sampling_hz = None

        if time_available and time_col:
            raw_time = df[time_col]
            try:
                # Try parsing as datetime
                parsed_time = pd.to_datetime(raw_time)
                chronological_coverage["start"] = str(parsed_time.min())
                chronological_coverage["end"] = str(parsed_time.max())
                time_diffs = parsed_time.diff().dropna().dt.total_seconds()
                median_dt = float(time_diffs.median()) if len(time_diffs) > 0 else None
                chronological_coverage["median_dt_seconds"] = median_dt
                if median_dt and median_dt > 0:
                    estimated_sampling_hz = round(1.0 / median_dt, 4)
                chronological_coverage["is_monotonic"] = bool(parsed_time.is_monotonic_increasing)
            except Exception:
                # If numeric seconds/hours
                if pd.api.types.is_numeric_dtype(raw_time):
                    chronological_coverage["start"] = float(raw_time.min())
                    chronological_coverage["end"] = float(raw_time.max())
                    diffs = raw_time.diff().dropna()
                    median_dt = float(diffs.median()) if len(diffs) > 0 else None
                    chronological_coverage["median_dt_numeric"] = median_dt
                    chronological_coverage["is_monotonic"] = bool(raw_time.is_monotonic_increasing)

        # Feature availability matrix evaluation
        feature_availability = self._evaluate_feature_availability(semantic_map, df)

        report = {
            "summary": {
                "file_name": file_name,
                "file_size_bytes": file_size_bytes,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "duplicate_rows": duplicate_rows,
                "number_of_batteries": num_batteries,
                "battery_identifiers": battery_ids[:10],  # truncated list for display
            },
            "canonical_mappings": semantic_map,
            "chronological_coverage": chronological_coverage,
            "estimated_sampling_hz": estimated_sampling_hz,
            "columns": column_details,
            "feature_availability": feature_availability,
        }

        return report

    def _evaluate_feature_availability(
        self, semantic_map: Dict[str, Optional[str]], df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Determines which features can be calculated or must be marked unavailable."""
        availability: Dict[str, Dict[str, Any]] = {}

        has_v = bool(semantic_map.get("voltage"))
        has_i = bool(semantic_map.get("current"))
        has_t = bool(semantic_map.get("temperature"))
        has_p = bool(semantic_map.get("power")) or (has_v and has_i)
        has_soc = bool(semantic_map.get("soc"))
        has_soh = bool(semantic_map.get("soh"))
        has_time = bool(semantic_map.get("timestamp"))
        has_cycle = bool(semantic_map.get("cycle"))
        has_cap = bool(semantic_map.get("capacity"))

        availability["electrical_features"] = {
            "supported": has_v and has_i,
            "status": "Available" if (has_v and has_i) else "Unavailable",
            "missing_inputs": [k for k, v in [("voltage", has_v), ("current", has_i)] if not v],
        }

        availability["thermal_features"] = {
            "supported": has_t,
            "status": "Available" if has_t else "Unavailable",
            "missing_inputs": [] if has_t else ["temperature"],
        }

        availability["temporal_features"] = {
            "supported": has_time,
            "status": "Available" if has_time else "Unavailable",
            "missing_inputs": [] if has_time else ["timestamp"],
        }

        availability["usage_soc_features"] = {
            "supported": has_soc,
            "status": "Available" if has_soc else "Unavailable",
            "missing_inputs": [] if has_soc else ["soc"],
        }

        availability["effective_resistance_proxy"] = {
            "supported": has_v and has_i,
            "status": "Available (Proxy dV/dI)" if (has_v and has_i) else "Unavailable",
            "missing_inputs": [k for k, v in [("voltage", has_v), ("current", has_i)] if not v],
            "note": "Named effective resistance proxy, not true internal resistance.",
        }

        availability["coulomb_counting_capacity"] = {
            "supported": has_i and has_time and has_cycle,
            "status": "Available" if (has_i and has_time and has_cycle) else "Unavailable",
            "missing_inputs": [k for k, v in [("current", has_i), ("timestamp", has_time), ("cycle", has_cycle)] if not v],
        }

        availability["equivalent_full_cycles"] = {
            "supported": has_soc or (has_i and has_time),
            "status": "Available" if (has_soc or (has_i and has_time)) else "Unavailable",
            "missing_inputs": [] if (has_soc or (has_i and has_time)) else ["soc or (current and timestamp)"],
        }

        availability["target_current_soh"] = {
            "supported": has_soh or has_cap,
            "status": "Available" if (has_soh or has_cap) else "Unavailable",
            "missing_inputs": [] if (has_soh or has_cap) else ["soh or capacity"],
        }

        return availability

    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Formats the inspection report into clean, professional Markdown."""
        s = report["summary"]
        cm = report["canonical_mappings"]
        cc = report["chronological_coverage"]
        fa = report["feature_availability"]

        lines = [
            "# ABIA Dataset Automated Inspection Report",
            "",
            "## 1. Executive Summary",
            f"- **File Name:** `{s['file_name']}`",
            f"- **File Size:** {s['file_size_bytes'] / 1024:.2f} KB",
            f"- **Total Samples:** {s['total_rows']:,}",
            f"- **Total Columns:** {s['total_columns']}",
            f"- **Duplicate Rows:** {s['duplicate_rows']:,}",
            f"- **Batteries / Modules Identified:** {s['number_of_batteries']}",
            f"- **Battery IDs:** {', '.join(s['battery_identifiers'])}",
            "",
            "## 2. Inferred Canonical Semantics",
            "| Physical Semantic | Mapped Column Name | Status |",
            "| :--- | :--- | :--- |",
        ]

        for semantic, col in cm.items():
            status = f"`{col}`" if col else "**MISSING / UNAVAILABLE**"
            lines.append(f"| {semantic.capitalize()} | {status} | {'Found' if col else 'Not in dataset'} |")

        lines.extend([
            "",
            "## 3. Chronological Coverage",
            f"- **Timestamp Available:** {cc.get('available')}",
            f"- **Coverage Interval:** {cc.get('start')} to {cc.get('end')}",
            f"- **Median Δt:** {cc.get('median_dt_seconds') or cc.get('median_dt_numeric')} seconds",
            f"- **Estimated Sampling Frequency:** {report.get('estimated_sampling_hz')} Hz",
            f"- **Strictly Monotonic Time:** {cc.get('is_monotonic')}",
            "",
            "## 4. Feature Availability Matrix",
            "| Feature Category | Support Status | Missing Prerequisites | Notes |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for feat_name, feat_info in fa.items():
            missing = ", ".join(feat_info.get("missing_inputs", [])) or "None"
            note = feat_info.get("note", "Standard")
            lines.append(f"| {feat_name.replace('_', ' ').title()} | {feat_info['status']} | {missing} | {note} |")

        lines.extend([
            "",
            "## 5. Column Diagnostics & Quality",
            "| Column | Dtype | Nulls (%) | Min | Max | Mean | Std | Infinite |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for col, col_info in report["columns"].items():
            cmin = f"{col_info.get('min'):.3f}" if col_info.get("min") is not None else "-"
            cmax = f"{col_info.get('max'):.3f}" if col_info.get("max") is not None else "-"
            cmean = f"{col_info.get('mean'):.3f}" if col_info.get("mean") is not None else "-"
            cstd = f"{col_info.get('std'):.3f}" if col_info.get("std") is not None else "-"
            inf_cnt = col_info.get("infinite_count", 0)
            lines.append(
                f"| `{col}` | {col_info['dtype']} | {col_info['null_percentage']}% | {cmin} | {cmax} | {cmean} | {cstd} | {inf_cnt} |"
            )

        return "\n".join(lines)
