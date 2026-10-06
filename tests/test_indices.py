"""
Tests for Phase 7 & 8: Health Indices and ABHI Engine.
"""

import numpy as np
import pandas as pd
import pytest
from src.abia.indices.health_indices import compute_all_health_indices
from src.abia.indices.abhi import ABHIEngine


def test_health_indices_bounded_0_to_100():
    n = 50
    df = pd.DataFrame({
        "voltage": np.linspace(4.2, 2.8, n),
        "current": np.linspace(5.0, 45.0, n),
        "power": np.linspace(20.0, 150.0, n),
        "temperature": np.linspace(25.0, 55.0, n),
        "soh": np.linspace(100.0, 70.0, n),
        "capacity_ah": np.linspace(50.0, 35.0, n),
        "effective_resistance_proxy": np.linspace(0.015, 0.045, n),
    })

    indices_df = compute_all_health_indices(df)

    for col in ["VSI", "TSI", "CSI", "PSI", "DGI"]:
        assert col in indices_df.columns
        vals = indices_df[col].dropna()
        assert (vals >= 0.0).all(), f"{col} has values below 0"
        assert (vals <= 100.0).all(), f"{col} has values above 100"


def test_abhi_weight_learning_and_constraints():
    n = 100
    np.random.seed(42)
    indices_df = pd.DataFrame({
        "VSI": np.random.uniform(70, 95, n),
        "TSI": np.random.uniform(65, 90, n),
        "CSI": np.random.uniform(60, 85, n),
        "PSI": np.random.uniform(65, 88, n),
        "DGI": np.random.uniform(70, 98, n),
    })
    target_soh = 0.5 * indices_df["DGI"] + 0.3 * indices_df["VSI"] + 0.2 * indices_df["TSI"]

    engine = ABHIEngine(min_weight=0.05)
    learned_w = engine.fit_weights(indices_df, target_soh)

    # Assert sum of weights equals 1
    assert np.isclose(np.sum(learned_w), 1.0)
    # Assert minimum weight constraint respected
    assert (learned_w >= 0.05 - 1e-4).all()

    # Assert ABHI bounded in 0-100
    abhi_series = engine.compute_abhi(indices_df, method="learned_constrained")
    assert (abhi_series >= 0.0).all()
    assert (abhi_series <= 100.0).all()

    # Compare formulations
    comp = engine.compare_weighting_formulations(indices_df, target_soh)
    assert "equal_weights" in comp
    assert "learned_constrained" in comp
    assert "context_aware" in comp
