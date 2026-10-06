"""
Domain Interaction Features for ABIA.
Calculates battery electro-thermal interactions and state responses.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_interaction_features(
    df: pd.DataFrame,
    v_col: str = "voltage",
    i_col: str = "current",
    t_col: str = "temperature",
    soc_col: str = "soc",
    p_col: Optional[str] = "power",
) -> pd.DataFrame:
    """
    Computes domain-specific electro-thermal interaction features.
    """
    out = pd.DataFrame(index=df.index)

    has_v = v_col in df.columns
    has_i = i_col in df.columns
    has_t = t_col in df.columns
    has_soc = soc_col in df.columns

    # 1. Voltage x Current (Instantaneous Power)
    if has_v and has_i:
        v = df[v_col]
        i = df[i_col]
        out["inter_V_x_I"] = v * i

        # Dynamic Voltage Response to Current: dV / dI
        dv = v.diff().fillna(0.0)
        di = i.diff().replace(0.0, np.nan)
        out["response_V_to_I"] = (dv / di).clip(-5.0, 5.0).fillna(0.0)

    # 2. Current x Temperature (Thermal-Joule Coupling)
    if has_i and has_t:
        i = df[i_col]
        t = df[t_col]
        out["inter_I_x_T"] = i * t
        # Joule heating proxy: I^2 * T
        out["inter_I2_x_T"] = (i**2) * t

    # 3. Power x Temperature (Heat Dissipation Stress)
    if has_t:
        t = df[t_col]
        p = df[p_col] if (p_col and p_col in df.columns) else (df[v_col] * df[i_col] if (has_v and has_i) else None)
        if p is not None:
            out["inter_P_x_T"] = p * t

    # 4. Voltage x SOC (State Trajectory Interaction)
    if has_v and has_soc:
        v = df[v_col]
        soc = df[soc_col]
        out["inter_V_x_SOC"] = v * (soc / 100.0)

        # Dynamic Voltage Response to SOC: dV / dSOC
        dsoc = soc.diff().replace(0.0, np.nan)
        dv = v.diff().fillna(0.0)
        out["response_V_to_SOC"] = (dv / dsoc).clip(-10.0, 10.0).fillna(0.0)

    # 5. Temperature x SOC (Thermal-State Coupling)
    if has_t and has_soc:
        t = df[t_col]
        soc = df[soc_col]
        out["inter_T_x_SOC"] = t * (soc / 100.0)

    return out.ffill().fillna(0.0)
