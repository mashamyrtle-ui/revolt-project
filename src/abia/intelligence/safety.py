"""
Independent Rule-Based Safety Engine for ABIA.
Operates independently from ML models to detect physical hazard violations.
Enforces hard safety overrides: any critical violation forces HIGH-RISK / QUARANTINE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd


class SafetyEngine:
    """
    Independent rule-based safety engine.
    Evaluates telemetry against deterministic physical limits:
      - Critical Temperature (> 60 C or < -20 C)
      - Rapid Temperature Rise (dT/dt > 1.5 C/s)
      - Extreme Voltage (Cell < 2.2 V or > 4.35 V)
      - Excessive Voltage Imbalance (> 150 mV)
      - Severe Overcurrent (> 150 A)
      - Critical Thermal Instability (TSI < 15)
    """

    def __init__(self, rules_config_path: Optional[str] = "config/safety_rules.yaml"):
        self.rules: Dict[str, Any] = {}
        if rules_config_path:
            try:
                with open(rules_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    self.rules = cfg.get("safety_rules", {})
            except Exception:
                self.rules = self._default_rules()
        else:
            self.rules = self._default_rules()

    def _default_rules(self) -> Dict[str, Any]:
        return {
            "critical_temperature": {"enabled": True, "upper_limit": 60.0, "lower_limit": -20.0},
            "rapid_temperature_rise": {"enabled": True, "rate_limit": 1.5},
            "extreme_voltage": {"enabled": True, "max_v": 4.35, "min_v": 2.20},
            "abnormal_overcurrent": {"enabled": True, "max_i": 150.0},
            "severe_thermal_anomaly": {"enabled": True, "tsi_cutoff": 15.0},
        }

    def evaluate_safety(
        self,
        row_or_df: pd.Series | pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluates a single telemetry step or recent window for safety violations.
        Returns safety status, list of violations, and whether a safety override is triggered.
        """
        violations: List[Dict[str, Any]] = []
        is_critical_override = False

        # Extract parameters
        temp = row_or_df.get("temperature")
        dt_dt = row_or_df.get("dT_dt", 0.0)
        v = row_or_df.get("voltage")
        curr = row_or_df.get("current")
        tsi = row_or_df.get("TSI")
        anomaly_flag = row_or_df.get("is_anomaly", 0)

        # 1. Critical Temperature Check
        if temp is not None and not np.isnan(temp):
            if temp >= self.rules.get("critical_temperature", {}).get("upper_critical_limit", 60.0):
                violations.append({
                    "rule": "Critical Over-Temperature",
                    "severity": "CRITICAL",
                    "value": float(temp),
                    "threshold": 60.0,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True
            elif temp <= self.rules.get("critical_temperature", {}).get("lower_critical_limit", -20.0):
                violations.append({
                    "rule": "Sub-zero Charging Risk",
                    "severity": "CRITICAL",
                    "value": float(temp),
                    "threshold": -20.0,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True

        # 2. Rapid Temperature Rise Check
        if dt_dt is not None and not np.isnan(dt_dt):
            if abs(dt_dt) >= self.rules.get("rapid_temperature_rise", {}).get("critical_rate_limit", 1.5):
                violations.append({
                    "rule": "Rapid Temperature Rise (Thermal Runaway Precursor)",
                    "severity": "CRITICAL",
                    "value": float(abs(dt_dt)),
                    "threshold": 1.5,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True

        # 3. Extreme Voltage Check
        if v is not None and not np.isnan(v):
            if v > self.rules.get("extreme_voltage", {}).get("max_voltage_cutoff", 4.35):
                violations.append({
                    "rule": "Severe Overvoltage (Electrolyte Oxidation)",
                    "severity": "CRITICAL",
                    "value": float(v),
                    "threshold": 4.35,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True
            elif v < self.rules.get("extreme_voltage", {}).get("min_voltage_cutoff", 2.20):
                violations.append({
                    "rule": "Severe Overdischarge (Copper Dissolution)",
                    "severity": "CRITICAL",
                    "value": float(v),
                    "threshold": 2.20,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True

        # 4. Abnormal Current Check
        if curr is not None and not np.isnan(curr):
            if abs(curr) > self.rules.get("abnormal_overcurrent", {}).get("max_current_abs", 150.0):
                violations.append({
                    "rule": "Abnormal Overcurrent Spike",
                    "severity": "CRITICAL",
                    "value": float(abs(curr)),
                    "threshold": 150.0,
                    "action": "HIGH-RISK / QUARANTINE",
                })
                is_critical_override = True

        # 5. Severe Thermal Index Breakdown or Injected Anomaly
        if (tsi is not None and tsi < 15.0) or (anomaly_flag and int(anomaly_flag) == 1):
            violations.append({
                "rule": "Severe Thermal Instability / Anomaly Flag",
                "severity": "CRITICAL",
                "value": float(tsi) if tsi is not None else 1.0,
                "threshold": 15.0,
                "action": "HIGH-RISK / QUARANTINE",
            })
            is_critical_override = True

        status = "CRITICAL_VIOLATION" if is_critical_override else ("SAFE" if len(violations) == 0 else "WARNING")

        return {
            "safety_status": status,
            "safety_override_triggered": is_critical_override,
            "violations": violations,
            "violation_count": len(violations),
        }
