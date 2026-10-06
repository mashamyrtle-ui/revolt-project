"""
Temporal Feature Engineering for ABIA.
Calculates multi-scale causal rolling trends, linear slopes, and trend persistence.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _calculate_causal_slope(series: pd.Series, window: int) -> pd.Series:
    """Calculates rolling linear regression slope over historical window using fast 1D convolution."""
    if len(series) < window:
        return pd.Series(0.0, index=series.index)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = float(np.sum((x - x_mean) ** 2))
    if x_var <= 0:
        return pd.Series(0.0, index=series.index)
    weights = (x - x_mean) / x_var
    y = np.nan_to_num(series.values.astype(float), nan=0.0)
    slopes_valid = np.convolve(y, weights[::-1], mode="valid")
    full_slopes = np.concatenate([np.zeros(window - 1), slopes_valid])
    return pd.Series(full_slopes, index=series.index)


def compute_temporal_features(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Computes multi-scale temporal statistics, slopes, and trend persistence.
    Windows:
      - short: 10 steps
      - medium: 50 steps
      - long: 100 steps
    """
    if windows is None:
        windows = {"short": 10, "medium": 50, "long": 100}

    if target_columns is None:
        # Default to continuous sensor channels if present
        candidates = ["voltage", "temperature", "current"]
        target_columns = [c for c in candidates if c in df.columns]

    out = pd.DataFrame(index=df.index)

    for col in target_columns:
        s = df[col]
        for w_name, w_size in windows.items():
            # Rolling statistics
            r = s.rolling(window=w_size, min_periods=2)
            mean_val = r.mean()
            std_val = r.std().fillna(0.0)

            out[f"{col}_roll_mean_{w_name}"] = mean_val
            out[f"{col}_roll_std_{w_name}"] = std_val

            # Net rate of change over window
            out[f"{col}_roc_{w_name}"] = (s - s.shift(w_size)).fillna(0.0)

            # Trend persistence: net directional change / sum of absolute step-wise changes
            abs_diff_sum = s.diff().abs().rolling(window=w_size, min_periods=2).sum()
            net_change = (s - s.shift(w_size)).abs()
            persistence = (net_change / abs_diff_sum.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
            out[f"{col}_persistence_{w_name}"] = persistence

            # Rolling linear slope
            if w_size <= 50:
                out[f"{col}_slope_{w_name}"] = _calculate_causal_slope(s, w_size)

    return out.ffill().fillna(0.0)
