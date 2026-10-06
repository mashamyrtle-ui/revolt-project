"""
Thermal Feature Engineering for ABIA.
Quantifies thermal rates, rolling stability, thermal trends, and operating stress.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_thermal_features(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    window_sizes: Optional[List[int]] = None,
    ambient_temp_ref: float = 25.0,
) -> pd.DataFrame:
    """
    Computes causal thermal dynamics.
    Strictly past-looking without future leakage.
    """
    if window_sizes is None:
        window_sizes = [10, 50]

    out = pd.DataFrame(index=df.index)

    if temp_col not in df.columns:
        return out

    temp = df[temp_col]

    # 1. Thermal rate of change
    out["dT_dt"] = temp.diff().fillna(0.0)

    # 2. Temperature deviation from reference ambient
    out["temp_dev_ambient"] = temp - ambient_temp_ref

    # 3. Rolling thermal statistics
    for w in window_sizes:
        r_t = temp.rolling(window=w, min_periods=2)
        out[f"temp_mean_w{w}"] = r_t.mean()
        out[f"temp_std_w{w}"] = r_t.std().fillna(0.0)
        out[f"temp_range_w{w}"] = (r_t.max() - r_t.min()).fillna(0.0)
        out[f"temp_variance_w{w}"] = r_t.var().fillna(0.0)
        out[f"temp_dev_w{w}"] = temp - out[f"temp_mean_w{w}"]

        # Thermal trend (difference over window span)
        out[f"temp_trend_w{w}"] = (temp - temp.shift(w)).fillna(0.0)

    # 4. Thermal stress indicators
    # Excess heat penalty above optimal Li-ion window (20°C - 35°C)
    out["thermal_stress_high"] = np.maximum(0.0, temp - 35.0)
    out["thermal_stress_low"] = np.maximum(0.0, 15.0 - temp)
    out["thermal_stress_composite"] = out["thermal_stress_high"] + 0.5 * out["thermal_stress_low"]

    return out.ffill().fillna(0.0)
