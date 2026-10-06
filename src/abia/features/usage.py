"""
Usage Feature Engineering for ABIA.
Quantifies cycling depth, throughput, C-rate, and operating stress metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_usage_features(
    df: pd.DataFrame,
    soc_col: Optional[str] = "soc",
    current_col: Optional[str] = "current",
    power_col: Optional[str] = "power",
    rated_capacity_ah: Optional[float] = 50.0,
    window_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Computes causal usage, stress, and throughput metrics.
    C-rate is calculated strictly when rated capacity is explicitly specified or estimated.
    """
    if window_sizes is None:
        window_sizes = [10, 50]

    out = pd.DataFrame(index=df.index)

    # 1. SOC-based features
    if soc_col and soc_col in df.columns:
        soc = df[soc_col]
        out["dSOC_dt"] = soc.diff().fillna(0.0)
        # Depth of Discharge
        out["dod"] = 100.0 - soc

        for w in window_sizes:
            r_soc = soc.rolling(window=w, min_periods=2)
            out[f"soc_std_w{w}"] = r_soc.std().fillna(0.0)
            out[f"soc_range_w{w}"] = (r_soc.max() - r_soc.min()).fillna(0.0)

        # Equivalent Full Cycles (EFC) proxy from SOC throughput: sum(|dSOC|) / 100
        out["cumulative_efc_proxy"] = out["dSOC_dt"].abs().cumsum() / 100.0
    else:
        out["dod_unavailable"] = True

    # 2. Current stress & C-rate
    if current_col and current_col in df.columns:
        current = df[current_col]
        if rated_capacity_ah and rated_capacity_ah > 0:
            # C-rate = I / Q_nominal
            out["c_rate"] = current / rated_capacity_ah
            out["c_rate_abs"] = current.abs() / rated_capacity_ah
        else:
            out["c_rate_status"] = "Unavailable: rated capacity unknown"

        # Current Stress: ratio to 95th percentile current in past observations
        # Using expanding percentile to strictly prevent future leakage
        cum_max_i = current.abs().expanding(min_periods=10).quantile(0.95).replace(0.0, 1.0)
        out["current_stress"] = (current.abs() / cum_max_i).clip(0.0, 3.0).fillna(0.0)

    # 3. Power stress
    if power_col and power_col in df.columns:
        power = df[power_col]
        cum_max_p = power.abs().expanding(min_periods=10).quantile(0.95).replace(0.0, 1.0)
        out["power_stress"] = (power.abs() / cum_max_p).clip(0.0, 3.0).fillna(0.0)

    return out.ffill().fillna(0.0)
