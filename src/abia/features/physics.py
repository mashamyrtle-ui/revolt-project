"""
Battery Physics Proxy Features for ABIA.
Calculates Effective Resistance Proxy (not true internal resistance),
Coulomb-counting capacity estimation, and Equivalent Full Cycles (EFC).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_physics_features(
    df: pd.DataFrame,
    v_col: str = "voltage",
    i_col: str = "current",
    time_col: Optional[str] = "timestamp",
    cycle_col: Optional[str] = "cycle_number",
    soc_col: Optional[str] = "soc",
    nominal_capacity_ah: float = 50.0,
    min_current_delta_for_reff: float = 1.0,
) -> pd.DataFrame:
    """
    Computes physics proxy features adhering strictly to scientific terminology.
    """
    out = pd.DataFrame(index=df.index)

    # 1. Effective Resistance Proxy: R_eff ≈ |ΔV / ΔI|
    # Strictly termed "effective resistance proxy"
    if v_col in df.columns and i_col in df.columns:
        dv = df[v_col].diff().fillna(0.0)
        di = df[i_col].diff().fillna(0.0)

        # Condition for valid excitation step: |di| >= threshold
        valid_excitation = di.abs() >= min_current_delta_for_reff
        raw_r_eff = (dv / di).abs()
        # Bound to physically reasonable lithium-ion module resistance range (e.g., 0.001 to 0.500 Ohms)
        bounded_r_eff = raw_r_eff.where(valid_excitation & (raw_r_eff > 0.001) & (raw_r_eff < 0.500), np.nan)

        # Causal forward fill of most recent valid effective resistance proxy
        out["effective_resistance_proxy"] = bounded_r_eff.ffill().fillna(0.020)
        # Resistance growth proxy relative to initial baseline
        initial_r = out["effective_resistance_proxy"].iloc[0] if len(out) > 0 else 0.020
        out["resistance_growth_ratio"] = out["effective_resistance_proxy"] / max(initial_r, 1e-4)

    # 2. Equivalent Full Cycles (EFC)
    if soc_col and soc_col in df.columns:
        # Sum of absolute SOC changes divided by 200 (100% discharge + 100% charge = 1 full cycle)
        abs_dsoc = df[soc_col].diff().abs().fillna(0.0)
        out["equivalent_full_cycles"] = abs_dsoc.cumsum() / 200.0
    elif i_col in df.columns and time_col and time_col in df.columns:
        # Fallback via ampere-hour throughput
        try:
            dt_s = pd.to_datetime(df[time_col]).diff().dt.total_seconds().fillna(10.0)
            ah_step = (df[i_col].abs() * dt_s) / 3600.0
            out["equivalent_full_cycles"] = ah_step.cumsum() / (2.0 * nominal_capacity_ah)
        except Exception:
            out["equivalent_full_cycles"] = 0.0

    # 3. Capacity estimation via Coulomb counting per cycle (if cycle column exists)
    if cycle_col and cycle_col in df.columns and i_col in df.columns:
        # Per cycle integrated discharge capacity: Q ≈ ∫ I dt
        # For discharges (positive current convention here or vice versa)
        if time_col and time_col in df.columns:
            try:
                dt_s = pd.to_datetime(df[time_col]).diff().dt.total_seconds().fillna(10.0)
            except Exception:
                dt_s = pd.Series(10.0, index=df.index)
        else:
            dt_s = pd.Series(10.0, index=df.index)

        ah_step = (df[i_col].clip(lower=0.0) * dt_s) / 3600.0
        # Expanding cycle cumulative capacity
        out["integrated_capacity_proxy_ah"] = ah_step.groupby(df[cycle_col]).cumsum()
        # Normalized capacity retention proxy
        out["capacity_retention_proxy"] = (out["integrated_capacity_proxy_ah"] / nominal_capacity_ah).clip(0.0, 1.2)

    return out.ffill().fillna(0.0)
