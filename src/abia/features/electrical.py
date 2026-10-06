"""
Electrical Feature Engineering for ABIA.
Calculates causal voltage, current, and power variations and response metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_electrical_features(
    df: pd.DataFrame,
    v_col: str = "voltage",
    i_col: str = "current",
    p_col: Optional[str] = "power",
    soc_col: Optional[str] = "soc",
    temp_col: Optional[str] = "temperature",
    window_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Computes causal electrical dynamics per battery.
    All rolling windows look backwards only (closed='right') to eliminate future leakage.
    """
    if window_sizes is None:
        window_sizes = [10, 50]

    out = pd.DataFrame(index=df.index)

    if v_col not in df.columns or i_col not in df.columns:
        return out

    v = df[v_col]
    i = df[i_col]
    p = df[p_col] if (p_col and p_col in df.columns) else (v * i)

    # 1. Differential features (Rates of change)
    out["dV_dt"] = v.diff().fillna(0.0)
    out["dI_dt"] = i.diff().fillna(0.0)

    # 2. Rolling statistics across specified window sizes
    for w in window_sizes:
        r_v = v.rolling(window=w, min_periods=2)
        r_i = i.rolling(window=w, min_periods=2)
        r_p = p.rolling(window=w, min_periods=2)

        out[f"v_mean_w{w}"] = r_v.mean()
        out[f"v_std_w{w}"] = r_v.std().fillna(0.0)
        out[f"v_range_w{w}"] = (r_v.max() - r_v.min()).fillna(0.0)
        out[f"v_variance_w{w}"] = r_v.var().fillna(0.0)
        out[f"v_dev_w{w}"] = v - out[f"v_mean_w{w}"]

        out[f"i_mean_w{w}"] = r_i.mean()
        out[f"i_std_w{w}"] = r_i.std().fillna(0.0)
        out[f"i_rms_w{w}"] = np.sqrt((i**2).rolling(window=w, min_periods=2).mean()).fillna(0.0)

        out[f"p_std_w{w}"] = r_p.std().fillna(0.0)

    # 3. Modeled Expected Voltage: V_expected = f(SOC, current, temperature)
    # Uses empirical Li-ion OCV model if SOC is available, otherwise rolling baseline
    if soc_col and soc_col in df.columns:
        s = df[soc_col].clip(0.1, 99.9) / 100.0
        # Polynomial OCV representation
        v_ocv = (
            3.14
            + 0.95 * s
            - 0.40 * (s**2)
            + 0.85 * (s**3)
            - 0.55 * (s**4)
            + 0.22 * np.log(s + 1e-4)
            - 0.05 * np.log(1.001 - s)
        )
        # Approximate internal resistance baseline R0 = 0.02 Ohm
        r0 = 0.02
        v_expected = v_ocv - (i * r0)
        out["v_expected"] = v_expected
        out["v_deviation_from_expected"] = v - v_expected
    else:
        # Fallback if SOC unavailable: expected voltage modeled as short-term rolling mean
        out["v_expected"] = out.get("v_mean_w10", v)
        out["v_deviation_from_expected"] = v - out["v_expected"]

    # 4. Voltage-Current dynamic relationship proxy
    di_clean = out["dI_dt"].replace(0.0, np.nan)
    out["dV_over_dI"] = (out["dV_dt"] / di_clean).clip(-5.0, 5.0).fillna(0.0)

    # Clean any residual NaNs causally
    return out.ffill().fillna(0.0)
