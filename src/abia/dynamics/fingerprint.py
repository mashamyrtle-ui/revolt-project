"""
Multidimensional Battery Behaviour Fingerprint.
Profiles a battery's unique operational personality across 7 standardized dimensions:
  1. Voltage Stability
  2. Thermal Stability
  3. Current Stress
  4. Power Stability
  5. Degradation Behaviour
  6. Anomaly Behaviour
  7. Cycling Behaviour
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BatteryBehaviourFingerprint:
    """Calculates and encapsulates a battery's 7-dimensional behavioral signature."""

    DIMENSIONS = [
        "voltage_stability",
        "thermal_stability",
        "current_stress",
        "power_stability",
        "degradation_behaviour",
        "anomaly_behaviour",
        "cycling_behaviour",
    ]

    def compute_fingerprint(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, float]:
        """
        Computes 7-dimensional behavioral fingerprint summary (each 0 to 100).
        Uses recent historical window (or full battery history).
        """
        recent = df.tail(window) if len(df) > window else df

        # 1. Voltage Stability (from VSI or voltage variance)
        vsi_val = float(recent["VSI"].mean()) if "VSI" in recent.columns else 85.0

        # 2. Thermal Stability (from TSI or temperature range)
        tsi_val = float(recent["TSI"].mean()) if "TSI" in recent.columns else 80.0

        # 3. Current Stress (from CSI)
        csi_val = float(recent["CSI"].mean()) if "CSI" in recent.columns else 75.0

        # 4. Power Stability (from PSI)
        psi_val = float(recent["PSI"].mean()) if "PSI" in recent.columns else 80.0

        # 5. Degradation Behaviour (from DGI)
        dgi_val = float(recent["DGI"].mean()) if "DGI" in recent.columns else 80.0

        # 6. Anomaly Behaviour (0 = frequent anomalies, 100 = completely clean)
        if "is_anomaly" in recent.columns:
            anomaly_rate = float(recent["is_anomaly"].mean())
            anomaly_score = max(0.0, 100.0 * (1.0 - 5.0 * anomaly_rate))
        else:
            anomaly_score = 95.0

        # 7. Cycling Behaviour (Maturity & stress intensity)
        # Moderate cycles with reasonable throughput score high
        if "equivalent_full_cycles" in recent.columns:
            efc = float(recent["equivalent_full_cycles"].max())
            # Normalize against typical 1000 cycle lifetime benchmark
            cycling_score = float(np.clip(100.0 - (efc / 10.0), 10.0, 100.0))
        elif "c_rate" in recent.columns:
            mean_c = float(recent["c_rate"].abs().mean())
            cycling_score = float(np.clip(100.0 - mean_c * 50.0, 10.0, 100.0))
        else:
            cycling_score = 75.0

        fingerprint = {
            "voltage_stability": round(float(np.clip(vsi_val, 0.0, 100.0)), 1),
            "thermal_stability": round(float(np.clip(tsi_val, 0.0, 100.0)), 1),
            "current_stress": round(float(np.clip(csi_val, 0.0, 100.0)), 1),
            "power_stability": round(float(np.clip(psi_val, 0.0, 100.0)), 1),
            "degradation_behaviour": round(float(np.clip(dgi_val, 0.0, 100.0)), 1),
            "anomaly_behaviour": round(float(np.clip(anomaly_score, 0.0, 100.0)), 1),
            "cycling_behaviour": round(float(np.clip(cycling_score, 0.0, 100.0)), 1),
        }

        return fingerprint
