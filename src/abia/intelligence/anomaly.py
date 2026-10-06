"""
Anomaly & Risk Engine for ABIA.
Evaluates multi-factor operational risk across:
  - Safety violations
  - Model prediction residuals
  - Health index deterioration (TSI, VSI, CSI)
  - Abrupt parameter rates (dT/dt, dV/dt)
Returns categorical risk: LOW, MEDIUM, HIGH, CRITICAL with dominant risk drivers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class AnomalyRiskEngine:
    """Evaluates battery risk level and synthesizes underlying physical drivers."""

    def evaluate_risk(
        self,
        row_or_window: pd.Series | pd.DataFrame,
        safety_status: Dict[str, Any],
        model_residual: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes composite risk score (0-100), categorical risk level, and driver explanations.
        """
        if isinstance(row_or_window, pd.DataFrame):
            row = row_or_window.iloc[-1]
        else:
            row = row_or_window

        drivers: List[str] = []
        risk_score = 15.0  # Base nominal risk

        # 1. Critical Safety Override Check
        if safety_status.get("safety_override_triggered", False):
            for v in safety_status.get("violations", []):
                drivers.append(f"Safety Violation: {v['rule']} ({v.get('value')})")
            return {
                "risk_level": "CRITICAL",
                "risk_score": 100.0,
                "primary_drivers": drivers,
            }

        # 2. Thermal Stress Evaluation
        tsi = row.get("TSI", 85.0)
        temp = row.get("temperature", 25.0)
        dt_dt = row.get("dT_dt", 0.0)

        if tsi < 50.0 or temp > 48.0:
            risk_score += 35.0
            drivers.append(f"Elevated cell temperature ({temp:.1f} °C) causing thermal stress")
        elif tsi < 70.0:
            risk_score += 15.0
            drivers.append(f"Moderate thermal stress (TSI: {tsi:.1f})")

        if abs(dt_dt) > 0.8:
            risk_score += 20.0
            drivers.append(f"High thermal rate of change (|dT/dt| = {abs(dt_dt):.2f} °C/step)")

        # 3. Voltage Stability Evaluation
        vsi = row.get("VSI", 85.0)
        if vsi < 50.0:
            risk_score += 25.0
            drivers.append(f"Severe voltage instability (VSI: {vsi:.1f})")
        elif vsi < 70.0:
            risk_score += 10.0
            drivers.append(f"Moderate voltage fluctuation (VSI: {vsi:.1f})")

        # 4. Model Residual Anomaly
        if abs(model_residual) > 5.0:
            risk_score += 20.0
            drivers.append(f"Unusual model prediction discrepancy (Residual: {model_residual:.2f}%)")

        # 5. Injected or Detected Anomaly Flag
        if row.get("is_anomaly", 0) == 1:
            risk_score += 40.0
            drivers.append("Telemetry anomaly signature flagged by sensor monitor")

        # Map continuous risk score to categorical levels
        risk_score = float(np.clip(risk_score, 0.0, 100.0))

        if risk_score >= 80.0:
            risk_level = "CRITICAL"
        elif risk_score >= 55.0:
            risk_level = "HIGH"
        elif risk_score >= 30.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not drivers:
            drivers.append("Nominal operating telemetry within standard tolerances")

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 1),
            "primary_drivers": drivers,
        }
