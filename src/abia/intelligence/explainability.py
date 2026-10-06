"""
Explainability & Decision Attribution Engine for ABIA.
Synthesizes domain indices, physical stresses, model residuals,
and safety boundaries into clear, actionable technical rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class ExplainabilityEngine:
    """Produces structured explanations for SOH predictions and circularity decisions."""

    def format_explanation(
        self,
        predicted_soh: float,
        future_soh: float,
        abhi: float,
        indices: Dict[str, float],
        momentum_status: str,
        risk_info: Dict[str, Any],
        confidence: float,
        circularity_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Creates structured explainability payload.
        """
        # Feature impact ranking
        contributing_factors = []
        dgi = indices.get("DGI", 80.0)
        tsi = indices.get("TSI", 80.0)
        vsi = indices.get("VSI", 80.0)
        csi = indices.get("CSI", 80.0)

        if dgi < 75.0:
            contributing_factors.append(f"Capacity loss & internal resistance growth (DGI: {dgi:.1f})")
        if tsi < 70.0:
            contributing_factors.append(f"Elevated thermal stress history (TSI: {tsi:.1f})")
        if vsi < 70.0:
            contributing_factors.append(f"Voltage response deviations under dynamic loads (VSI: {vsi:.1f})")
        if csi < 70.0:
            contributing_factors.append(f"High-rate current throughput and pulsed stress (CSI: {csi:.1f})")

        if not contributing_factors:
            contributing_factors.append("Nominal electro-thermal aging with balanced stress profile")

        return {
            "predicted_soh": round(predicted_soh, 2),
            "future_soh": round(future_soh, 2),
            "abhi": round(abhi, 1),
            "health_indices": {k: round(v, 1) for k, v in indices.items()},
            "health_momentum": momentum_status,
            "risk_level": risk_info.get("risk_level", "LOW"),
            "risk_score": risk_info.get("risk_score", 15.0),
            "confidence_pct": round(confidence, 1),
            "circularity_decision": circularity_decision.get("decision"),
            "circularity_suitability_score": circularity_decision.get("css_score"),
            "major_contributing_factors": contributing_factors,
            "decision_reasons": circularity_decision.get("reasons", []),
        }
