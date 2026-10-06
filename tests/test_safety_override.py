"""
Tests for Phase 14 & 15: Safety Override and Circularity Decision Engine.
"""

import pandas as pd
import pytest
from src.abia.intelligence.safety import SafetyEngine
from src.abia.intelligence.circularity import CircularityEngine
from src.abia.intelligence.anomaly import AnomalyRiskEngine


def test_safety_override_triggers_quarantine_even_with_high_soh():
    """
    CRITICAL REQUIREMENT:
    Do not allow a high-risk battery to be classified as 'Continue Using' because of high SOH.
    """
    safety = SafetyEngine()
    circ = CircularityEngine()
    risk_engine = AnomalyRiskEngine()

    # Battery has 98% SOH (seemingly brand new), but cell temperature is 65°C (thermal runaway risk!)
    row = pd.Series({
        "temperature": 65.0,
        "dT_dt": 2.0,
        "voltage": 3.8,
        "current": 20.0,
        "TSI": 10.0,
        "VSI": 85.0,
        "CSI": 80.0,
        "PSI": 80.0,
        "DGI": 95.0,
    })

    safety_status = safety.evaluate_safety(row)
    assert safety_status["safety_override_triggered"] is True
    assert safety_status["safety_status"] == "CRITICAL_VIOLATION"

    risk_info = risk_engine.evaluate_risk(row, safety_status)
    assert risk_info["risk_level"] == "CRITICAL"

    decision_info = circ.evaluate_decision(
        current_soh=98.0,
        future_soh=97.0,
        abhi=90.0,
        degradation_rate=0.01,
        momentum_status="Stable",
        vsi=85.0,
        tsi=10.0,
        csi=80.0,
        psi=80.0,
        dgi=95.0,
        risk_info=risk_info,
        confidence=92.0,
        safety_status=safety_status,
    )

    # Must be strictly HIGH-RISK / QUARANTINE
    assert decision_info["decision"] == "HIGH-RISK / QUARANTINE"
    assert decision_info["is_safety_override"] is True


def test_all_six_circularity_decisions():
    circ = CircularityEngine()
    dummy_safety_ok = {"safety_override_triggered": False, "violations": []}
    dummy_low_risk = {"risk_level": "LOW", "risk_score": 10.0}

    # 1. CONTINUE USING
    d1 = circ.evaluate_decision(
        current_soh=92.0,
        future_soh=90.0,
        abhi=90.0,
        degradation_rate=0.01,
        momentum_status="Stable",
        vsi=90.0,
        tsi=88.0,
        csi=85.0,
        psi=85.0,
        dgi=92.0,
        risk_info=dummy_low_risk,
        confidence=95.0,
        safety_status=dummy_safety_ok,
    )
    assert d1["decision"] == "CONTINUE USING"

    # 2. NEEDS MONITORING (SOH 82%, in 80-85% zone)
    d2 = circ.evaluate_decision(
        current_soh=82.0,
        future_soh=80.0,
        abhi=80.0,
        degradation_rate=0.03,
        momentum_status="Rapidly Deteriorating",
        vsi=80.0,
        tsi=78.0,
        csi=75.0,
        psi=75.0,
        dgi=80.0,
        risk_info=dummy_low_risk,
        confidence=90.0,
        safety_status=dummy_safety_ok,
    )
    assert d2["decision"] == "NEEDS MONITORING"

    # 3. SECOND-LIFE SUITABLE (SOH 74%, stable thermal/voltage)
    d3 = circ.evaluate_decision(
        current_soh=74.0,
        future_soh=72.0,
        abhi=75.0,
        degradation_rate=0.02,
        momentum_status="Slowly Deteriorating",
        vsi=80.0,
        tsi=75.0,
        csi=70.0,
        psi=72.0,
        dgi=74.0,
        risk_info=dummy_low_risk,
        confidence=88.0,
        safety_status=dummy_safety_ok,
    )
    assert d3["decision"] == "SECOND-LIFE SUITABLE"

    # 4. RECYCLE (SOH 58% < 65%)
    d4 = circ.evaluate_decision(
        current_soh=58.0,
        future_soh=55.0,
        abhi=50.0,
        degradation_rate=0.06,
        momentum_status="Rapidly Deteriorating",
        vsi=60.0,
        tsi=60.0,
        csi=55.0,
        psi=55.0,
        dgi=55.0,
        risk_info=dummy_low_risk,
        confidence=85.0,
        safety_status=dummy_safety_ok,
    )
    assert d4["decision"] == "RECYCLE"

    # 5. REPAIR / SERVICE (High risk due to localized imbalance/elevated stress)
    d5 = circ.evaluate_decision(
        current_soh=86.0,
        future_soh=84.0,
        abhi=70.0,
        degradation_rate=0.02,
        momentum_status="Slowly Deteriorating",
        vsi=80.0,
        tsi=35.0,  # poor thermal stability
        csi=70.0,
        psi=70.0,
        dgi=85.0,
        risk_info={"risk_level": "HIGH", "risk_score": 65.0},
        confidence=82.0,
        safety_status=dummy_safety_ok,
    )
    assert d5["decision"] == "REPAIR / SERVICE"

    # 6. HIGH-RISK / QUARANTINE
    d6 = circ.evaluate_decision(
        current_soh=78.0,
        future_soh=76.0,
        abhi=65.0,
        degradation_rate=0.04,
        momentum_status="Rapidly Deteriorating",
        vsi=40.0,
        tsi=20.0,
        csi=60.0,
        psi=50.0,
        dgi=70.0,
        risk_info={"risk_level": "CRITICAL", "risk_score": 90.0},
        confidence=75.0,
        safety_status={"safety_override_triggered": True, "violations": [{"rule": "Thermal runaway", "value": 62.0}]},
    )
    assert d6["decision"] == "HIGH-RISK / QUARANTINE"
