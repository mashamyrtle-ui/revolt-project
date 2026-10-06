"""
Circularity Decision Engine & Circularity Suitability Score (CSS).
Implements a strict, explainable decision hierarchy with safety override guarantees.
Evaluates batteries for 6 definitive circularity pathways:
  1. CONTINUE USING
  2. NEEDS MONITORING
  3. REPAIR / SERVICE
  4. SECOND-LIFE SUITABLE
  5. RECYCLE
  6. HIGH-RISK / QUARANTINE
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import yaml


class CircularityEngine:
    """
    Evaluates battery end-of-first-life status and assigns one of 6 circularity decisions.
    Integrates the Circularity Suitability Score (CSS) and guarantees safety rule overrides.
    """

    VALID_DECISIONS = [
        "CONTINUE USING",
        "NEEDS MONITORING",
        "REPAIR / SERVICE",
        "SECOND-LIFE SUITABLE",
        "RECYCLE",
        "HIGH-RISK / QUARANTINE",
    ]

    def __init__(self, config_path: Optional[str] = "config/config.yaml"):
        # Research prototype thresholds (Configurable)
        self.soh_recycle_limit = 65.0
        self.soh_second_life_min = 65.0
        self.soh_second_life_max = 80.0
        self.soh_continue_use_min = 85.0
        self.tsi_second_life_min = 68.0
        self.vsi_second_life_min = 68.0
        self.css_recycle_max = 42.0

        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    circ_cfg = cfg.get("circularity", {}).get("thresholds", {})
                    self.soh_recycle_limit = circ_cfg.get("soh_recycle_limit", self.soh_recycle_limit)
                    self.soh_second_life_min = circ_cfg.get("soh_second_life_min", self.soh_second_life_min)
                    self.soh_second_life_max = circ_cfg.get("soh_second_life_max", self.soh_second_life_max)
                    self.soh_continue_use_min = circ_cfg.get("soh_continue_use_min", self.soh_continue_use_min)
                    self.tsi_second_life_min = circ_cfg.get("tsi_second_life_min", self.tsi_second_life_min)
                    self.vsi_second_life_min = circ_cfg.get("vsi_second_life_min", self.vsi_second_life_min)
                    self.css_recycle_max = circ_cfg.get("css_recycle_max", self.css_recycle_max)
            except Exception:
                pass

    def compute_css(
        self,
        current_soh: float,
        future_soh: float,
        abhi: float,
        risk_score: float,
        confidence: float,
    ) -> float:
        """
        Computes Circularity Suitability Score (CSS):
        CSS = 0.35 * SOH + 0.20 * Future_SOH + 0.15 * ABHI + 0.15 * (100 - Risk) + 0.15 * Confidence
        """
        w_soh = 0.35
        w_future = 0.20
        w_abhi = 0.15
        w_risk = 0.15
        w_conf = 0.15

        risk_inverted = max(0.0, 100.0 - risk_score)
        css = (
            w_soh * current_soh
            + w_future * future_soh
            + w_abhi * abhi
            + w_risk * risk_inverted
            + w_conf * confidence
        )
        return round(float(np.clip(css, 0.0, 100.0)), 1)

    def evaluate_decision(
        self,
        current_soh: float,
        future_soh: float,
        abhi: float,
        degradation_rate: float,
        momentum_status: str,
        vsi: float,
        tsi: float,
        csi: float,
        psi: float,
        dgi: float,
        risk_info: Dict[str, Any],
        confidence: float,
        safety_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hierarchical, explainable circularity decision logic.
        Guarantees: If safety override is triggered, decision is strictly HIGH-RISK / QUARANTINE.
        """
        risk_level = risk_info.get("risk_level", "LOW")
        risk_score = risk_info.get("risk_score", 15.0)
        safety_override = safety_status.get("safety_override_triggered", False)

        # 1. Compute Circularity Suitability Score (CSS)
        css = self.compute_css(current_soh, future_soh, abhi, risk_score, confidence)

        reasons: List[str] = []

        # ========================================================
        # LEVEL 1: CRITICAL SAFETY OVERRIDE (HIGHEST PRIORITY)
        # ========================================================
        if safety_override or risk_level == "CRITICAL":
            decision = "HIGH-RISK / QUARANTINE"
            reasons.append("Safety override triggered by critical threshold violation")
            for v in safety_status.get("violations", []):
                reasons.append(f"Safety violation: {v['rule']} (Value: {v.get('value')})")
            if not safety_status.get("violations") and risk_level == "CRITICAL":
                reasons.append("Critical operational risk score detected by anomaly engine")

            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": True,
            }

        # ========================================================
        # LEVEL 2: REPAIR / SERVICE
        # ========================================================
        if risk_level == "HIGH" or (tsi < 40.0 and current_soh >= self.soh_second_life_min):
            decision = "REPAIR / SERVICE"
            reasons.append("Elevated operational risk or thermal instability detected")
            reasons.append(f"Current SOH ({current_soh:.1f}%) warrants servicing before redeployment")
            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": False,
            }

        # ========================================================
        # LEVEL 3: RECYCLE (END OF LIFE)
        # ========================================================
        if current_soh < self.soh_recycle_limit or css < self.css_recycle_max:
            decision = "RECYCLE"
            reasons.append(f"Severe irreversible degradation: SOH ({current_soh:.1f}%) below recycling limit ({self.soh_recycle_limit}%)")
            reasons.append(f"Circularity Suitability Score ({css:.1f}) indicates exhausted electrochemical utility")
            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": False,
            }

        # ========================================================
        # LEVEL 4: SECOND-LIFE SUITABLE (STATIONARY ESS)
        # ========================================================
        if (
            self.soh_second_life_min <= current_soh < self.soh_second_life_max
            and tsi >= self.tsi_second_life_min
            and vsi >= self.vsi_second_life_min
            and risk_level in ["LOW", "MEDIUM"]
        ):
            decision = "SECOND-LIFE SUITABLE"
            reasons.append(f"SOH ({current_soh:.1f}%) ideal for stationary energy storage (65% - 80% window)")
            reasons.append(f"Stable thermal behaviour (TSI: {tsi:.1f}) and voltage dynamics (VSI: {vsi:.1f})")
            reasons.append("No active safety violations; low-stress second-life repurposing recommended")
            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": False,
            }

        # ========================================================
        # LEVEL 5: NEEDS MONITORING
        # ========================================================
        if (
            (self.soh_second_life_max <= current_soh < self.soh_continue_use_min)
            or momentum_status == "Rapidly Deteriorating"
            or risk_level == "MEDIUM"
        ):
            decision = "NEEDS MONITORING"
            if current_soh < self.soh_continue_use_min:
                reasons.append(f"SOH ({current_soh:.1f}%) in transition boundary (80% - 85%)")
            if momentum_status == "Rapidly Deteriorating":
                reasons.append("Health Momentum indicates accelerating degradation rate")
            if risk_level == "MEDIUM":
                reasons.append("Moderate operational stress requires telemetry tracking")
            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": False,
            }

        # ========================================================
        # LEVEL 6: CONTINUE USING (HEALTHY EV OPERATION)
        # ========================================================
        if current_soh >= self.soh_continue_use_min and risk_level == "LOW":
            decision = "CONTINUE USING"
            reasons.append(f"Healthy battery state: High SOH ({current_soh:.1f}%)")
            reasons.append(f"High comprehensive health index (ABHI: {abhi:.1f})")
            reasons.append(f"Stable health momentum ({momentum_status}) and low operating risk")
            return {
                "decision": decision,
                "css_score": css,
                "reasons": reasons,
                "is_safety_override": False,
            }

        # Default fallback if boundary case
        decision = "NEEDS MONITORING"
        reasons.append(f"Telemetry meets borderline conditions (SOH: {current_soh:.1f}%, CSS: {css:.1f})")
        return {
            "decision": decision,
            "css_score": css,
            "reasons": reasons,
            "is_safety_override": False,
        }
