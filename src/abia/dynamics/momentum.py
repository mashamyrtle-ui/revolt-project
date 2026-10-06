"""
Battery Health Momentum (BHM) Engine.
Classifies battery degradation trajectory into:
  - Stable
  - Slowly Deteriorating
  - Rapidly Deteriorating
  - Temporarily Fluctuating
Uses strictly historical SOH and stress trajectories with ZERO target leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class HealthMomentumEngine:
    """Calculates historical SOH slope, acceleration, and degradation momentum category."""

    def __init__(
        self,
        slope_window: int = 15,
        accel_window: int = 30,
        stable_threshold: float = 0.015,
        rapid_threshold: float = 0.08,
    ):
        self.slope_window = slope_window
        self.accel_window = accel_window
        self.stable_threshold = stable_threshold
        self.rapid_threshold = rapid_threshold

    def compute_momentum(
        self,
        df: pd.DataFrame,
        soh_col: str = "soh",
        cycle_col: Optional[str] = "cycle_number",
        battery_col: str = "battery_id",
    ) -> pd.DataFrame:
        """
        Computes causal historical SOH slope (d SOH / d cycle), acceleration,
        and categorical momentum label per battery.
        """
        out = pd.DataFrame(index=df.index)

        if soh_col not in df.columns:
            out["bhm_slope"] = 0.0
            out["bhm_acceleration"] = 0.0
            out["bhm_volatility"] = 0.0
            out["bhm_status"] = "Unavailable: no SOH history"
            return out

        if battery_col in df.columns:
            groups = df.groupby(battery_col, sort=False)
        else:
            groups = [("ALL", df)]

        slope_list = []
        acc_list = []
        vol_list = []
        status_list = []

        for _, group in groups:
            s = group[soh_col]

            # 1. Historical SOH Slope (Rate of loss per step/cycle)
            # Strictly looking backwards: (SOH[t] - SOH[t - window]) / window
            # Negative means degrading
            hist_diff = s - s.shift(self.slope_window)
            slope = (hist_diff / float(self.slope_window)).fillna(0.0)

            # 2. Historical Acceleration: d(Slope) / dt
            slope_diff = slope - slope.shift(self.slope_window)
            accel = (slope_diff / float(self.slope_window)).fillna(0.0)

            # 3. Volatility / Noise (rolling standard deviation of step-to-step diffs)
            step_diffs = s.diff().fillna(0.0)
            vol = step_diffs.rolling(window=self.slope_window, min_periods=2).std().fillna(0.0)

            # 4. Categorize momentum
            statuses = []
            for sl, ac, v in zip(slope, accel, vol):
                loss_rate = -sl  # positive is degradation rate
                if v > 0.35 and loss_rate < self.rapid_threshold:
                    cat = "Temporarily Fluctuating"
                elif loss_rate < self.stable_threshold:
                    cat = "Stable"
                elif loss_rate >= self.rapid_threshold or ac < -0.005:
                    cat = "Rapidly Deteriorating"
                else:
                    cat = "Slowly Deteriorating"
                statuses.append(cat)

            slope_list.extend(slope)
            acc_list.extend(accel)
            vol_list.extend(vol)
            status_list.extend(statuses)

        out["bhm_slope"] = slope_list
        out["bhm_acceleration"] = acc_list
        out["bhm_volatility"] = vol_list
        out["bhm_status"] = status_list

        return out
