"""
ABIA Battery Intelligence Report Generator.
Synthesizes end-to-end telemetry analytics into a structured, executive-grade intelligence report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class BatteryIntelligenceReportGenerator:
    """Generates standardized ABIA Battery Intelligence Reports in JSON and Markdown formats."""

    @staticmethod
    def generate_report(
        battery_id: str,
        current_soh: float,
        future_soh: float,
        abhi: float,
        indices: Dict[str, float],
        health_momentum: str,
        fingerprint: Dict[str, float],
        degradation_rate: float,
        risk_info: Dict[str, Any],
        confidence: float,
        safety_status: Dict[str, Any],
        circularity_decision: Dict[str, Any],
        output_dir: Optional[str | Path] = "reports",
    ) -> Dict[str, Any]:
        """Generates, saves, and returns the comprehensive ABIA intelligence report."""
        report = {
            "battery_id": battery_id,
            "current_soh": round(float(current_soh), 2),
            "future_soh": round(float(future_soh), 2),
            "abhi": round(float(abhi), 1),
            "vsi": round(float(indices.get("VSI", 0.0)), 1),
            "tsi": round(float(indices.get("TSI", 0.0)), 1),
            "csi": round(float(indices.get("CSI", 0.0)), 1),
            "psi": round(float(indices.get("PSI", 0.0)), 1),
            "dgi": round(float(indices.get("DGI", 0.0)), 1),
            "health_momentum": health_momentum,
            "behaviour_fingerprint": fingerprint,
            "degradation_rate_pct_per_100_cycles": round(float(degradation_rate), 3),
            "anomaly_risk": risk_info.get("risk_level", "LOW"),
            "risk_score": risk_info.get("risk_score", 15.0),
            "confidence_pct": round(float(confidence), 1),
            "circularity_suitability_score": circularity_decision.get("css_score", 0.0),
            "safety_status": safety_status.get("safety_status", "SAFE"),
            "safety_override_triggered": safety_status.get("safety_override_triggered", False),
            "final_circularity_decision": circularity_decision.get("decision", "NEEDS MONITORING"),
            "decision_explanation": circularity_decision.get("reasons", []),
            "primary_risk_drivers": risk_info.get("primary_drivers", []),
        }

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            # Save JSON
            json_file = out_p / f"ABIA_Intelligence_Report_{battery_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # Save Markdown
            md_file = out_p / f"ABIA_Intelligence_Report_{battery_id}.md"
            md_content = BatteryIntelligenceReportGenerator.format_markdown(report)
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_content)

        return report

    @staticmethod
    def format_markdown(r: Dict[str, Any]) -> str:
        """Formats the report dictionary into professional Markdown."""
        fp = r["behaviour_fingerprint"]
        fp_rows = "\n".join([f"| {k.replace('_', ' ').title()} | {v}/100 |" for k, v in fp.items()])
        reasons = "\n".join([f"- {reason}" for reason in r["decision_explanation"]])

        return f"""# AETHERIX BATTERY INTELLIGENCE REPORT
**System:** ABIA v1.0.0 (Aetherix Battery Intelligence Algorithm)  
**Target Battery Identifier:** `{r['battery_id']}`

---

## 1. Executive Status
| Metric | Value | Reference / Status |
| :--- | :--- | :--- |
| **Current SOH** | **{r['current_soh']}%** | Estimated via ABIA Domain-Informed Predictor |
| **Future SOH (Horizon)** | **{r['future_soh']}%** | Causal multi-horizon forecast |
| **Degradation Rate** | **{r['degradation_rate_pct_per_100_cycles']}%** | Loss per 100 equivalent full cycles |
| **Aetherix Health Index (ABHI)** | **{r['abhi']} / 100** | Learned constrained multi-index fusion |
| **Health Momentum** | **{r['health_momentum']}** | Historical trajectory curvature |
| **Anomaly Risk Level** | **{r['anomaly_risk']}** | Risk Score: {r['risk_score']}/100 |
| **Prediction Confidence** | **{r['confidence_pct']}%** | Distributional & epistemic grounded |
| **Safety Status** | **{r['safety_status']}** | Override Triggered: `{r['safety_override_triggered']}` |
| **Circularity Suitability (CSS)** | **{r['circularity_suitability_score']} / 100** | Multi-factor circularity readiness |
| **FINAL CIRCULARITY DECISION** | **`{r['final_circularity_decision']}`** | Certified via Decision Matrix |

---

## 2. Normalized Health Indices (0 - 100)
- **Voltage Stability Index (VSI):** {r['vsi']} / 100
- **Thermal Stress Index (TSI):** {r['tsi']} / 100
- **Current Stress Index (CSI):** {r['csi']} / 100
- **Power Stability Index (PSI):** {r['psi']} / 100
- **Degradation Index (DGI):** {r['dgi']} / 100

---

## 3. 7-Dimensional Battery Behaviour Fingerprint
| Behavioural Dimension | Score (0 - 100) |
| :--- | :--- |
{fp_rows}

---

## 4. Circularity Decision Rationale & Explainability
**Assigned Pathway:** `{r['final_circularity_decision']}`

### Key Justifications:
{reasons}

### Risk & Anomaly Assessment:
{chr(10).join([f"- {driver}" for driver in r['primary_risk_drivers']])}

---
*Disclaimer: Research prototype evaluation. Not an OEM/automotive certified safety standard.*
"""
