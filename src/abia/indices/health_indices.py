"""
Modular Health Indices for ABIA:
  - VSI: Voltage Stability Index (0-100)
  - TSI: Thermal Stress/Stability Index (0-100)
  - CSI: Current Stress Index (0-100)
  - PSI: Power Stability Index (0-100)
  - DGI: Degradation Index (0-100)
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_vsi(
    v_series: pd.Series,
    dv_dt_series: pd.Series,
    lambda_v: float = 3.0,
    gamma_v: float = 0.5,
    window: int = 20,
) -> pd.Series:
    """
    Voltage Stability Index (VSI):
    Normalized to 0-100. Smooth, stable voltage responses yield ~100.
    Formula: VSI = 100 * exp(- (lambda_v * (sigma_v / mu_v) + gamma_v * |dV/dt|))
    """
    r_v = v_series.rolling(window=window, min_periods=2)
    mu_v = r_v.mean().replace(0.0, np.nan)
    sigma_v = r_v.std().fillna(0.0)

    cov_v = (sigma_v / mu_v).fillna(0.0).clip(0.0, 1.0)
    abs_dv = dv_dt_series.abs().fillna(0.0).clip(0.0, 2.0)

    exponent = -(lambda_v * cov_v + gamma_v * abs_dv)
    vsi = 100.0 * np.exp(exponent)
    return vsi.clip(0.0, 100.0)


def compute_tsi(
    temp_series: pd.Series,
    dt_dt_series: pd.Series,
    t_opt_min: float = 15.0,
    t_opt_max: float = 35.0,
    t_crit_delta: float = 25.0,
    alpha_t: float = 1.5,
) -> pd.Series:
    """
    Thermal Stress / Stability Index (TSI):
    Normalized to 0-100. Temperatures within optimal Li-ion window (15-35 C) yield ~100.
    Excess heat and rapid heating rates dT/dt penalize the index towards 0.
    """
    high_penalty = np.maximum(0.0, temp_series - t_opt_max)
    low_penalty = 0.5 * np.maximum(0.0, t_opt_min - temp_series)
    rate_penalty = alpha_t * dt_dt_series.abs().fillna(0.0)

    total_stress = (high_penalty + low_penalty + rate_penalty) / t_crit_delta
    tsi = 100.0 * (1.0 - np.clip(total_stress, 0.0, 1.0))
    return tsi.clip(0.0, 100.0)


def compute_csi(
    i_series: pd.Series,
    i_stress_ref: Optional[float] = None,
    window: int = 20,
) -> pd.Series:
    """
    Current Stress Index (CSI):
    Normalized to 0-100. High continuous RMS loads and large current swings penalize the score.
    """
    i_rms = np.sqrt((i_series**2).rolling(window=window, min_periods=2).mean()).fillna(0.0)
    if i_stress_ref is None or i_stress_ref <= 0:
        i_stress_ref = float(i_series.abs().expanding(min_periods=10).quantile(0.95).max())
        i_stress_ref = max(10.0, i_stress_ref)

    stress_ratio = (i_rms / i_stress_ref).clip(0.0, 1.5)
    csi = 100.0 * (1.0 - np.clip(stress_ratio / 1.2, 0.0, 1.0))
    return csi.clip(0.0, 100.0)


def compute_psi(
    p_series: pd.Series,
    p_stress_ref: Optional[float] = None,
    window: int = 20,
) -> pd.Series:
    """
    Power Stability Index (PSI):
    Normalized to 0-100. Volatility in power draw penalizes the score.
    """
    p_std = p_series.rolling(window=window, min_periods=2).std().fillna(0.0)
    if p_stress_ref is None or p_stress_ref <= 0:
        p_stress_ref = float(p_series.abs().expanding(min_periods=10).quantile(0.95).max())
        p_stress_ref = max(20.0, p_stress_ref)

    power_volatility = (p_std / p_stress_ref).clip(0.0, 2.0)
    psi = 100.0 * (1.0 - np.clip(power_volatility, 0.0, 1.0))
    return psi.clip(0.0, 100.0)


def compute_dgi(
    df: pd.DataFrame,
    soh_col: Optional[str] = "soh",
    capacity_col: Optional[str] = "capacity_ah",
    reff_col: Optional[str] = "effective_resistance_proxy",
    nominal_capacity_ah: float = 50.0,
    theta_cap: float = 0.60,
) -> pd.Series:
    """
    Degradation Index (DGI):
    Normalized to 0-100. Quantifies irreversible capacity loss and resistance escalation.
    DGI = 100 * [ theta * (Q_retention) + (1 - theta) * (Resistance_Retention) ]
    """
    n = len(df)
    # Capacity retention term
    if soh_col and soh_col in df.columns:
        cap_term = (df[soh_col] / 100.0).clip(0.0, 1.0)
    elif capacity_col and capacity_col in df.columns:
        cap_term = (df[capacity_col] / nominal_capacity_ah).clip(0.0, 1.0)
    else:
        cap_term = pd.Series(1.0, index=df.index)

    # Resistance degradation term
    if reff_col and reff_col in df.columns:
        reff = df[reff_col]
        r_init = reff.iloc[0] if len(reff) > 0 else 0.02
        r_init = max(1e-4, r_init)
        # As resistance increases beyond 2.5x initial, resistance health approaches 0
        r_health = 1.0 - np.clip((reff - r_init) / (1.5 * r_init), 0.0, 1.0)
    else:
        r_health = pd.Series(1.0, index=df.index)

    dgi = 100.0 * (theta_cap * cap_term + (1.0 - theta_cap) * r_health)
    return dgi.clip(0.0, 100.0)


def compute_all_health_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all 5 normalized health indices."""
    out = pd.DataFrame(index=df.index)
    v_col = "voltage"
    i_col = "current"
    p_col = "power" if "power" in df.columns else None
    t_col = "temperature"
    dv_col = "dV_dt" if "dV_dt" in df.columns else None
    dt_col = "dT_dt" if "dT_dt" in df.columns else None

    # VSI
    dv_series = df[dv_col] if dv_col else df[v_col].diff().fillna(0.0)
    out["VSI"] = compute_vsi(df[v_col], dv_series)

    # TSI
    dt_series = df[dt_col] if dt_col else df[t_col].diff().fillna(0.0)
    out["TSI"] = compute_tsi(df[t_col], dt_series)

    # CSI
    out["CSI"] = compute_csi(df[i_col])

    # PSI
    p_series = df[p_col] if p_col else (df[v_col] * df[i_col])
    out["PSI"] = compute_psi(p_series)

    # DGI
    out["DGI"] = compute_dgi(df)

    return out
