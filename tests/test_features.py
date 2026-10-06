"""
Tests for Phases 3, 4, 5, 6: Feature Engineering Pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.abia.features.pipeline import FeaturePipeline


def test_feature_pipeline_execution():
    n = 60
    df = pd.DataFrame({
        "battery_id": ["BAT_TEST"] * n,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10s"),
        "cycle_number": [1] * 30 + [2] * 30,
        "cycle_step": list(range(1, 31)) * 2,
        "voltage": np.linspace(4.1, 3.2, n) + np.random.normal(0, 0.01, n),
        "current": [15.0] * n,
        "temperature": np.linspace(25.0, 32.0, n),
        "power": np.linspace(4.1, 3.2, n) * 15.0,
        "soc": np.linspace(95.0, 20.0, n),
        "soh": [98.0] * n,
    })

    pipeline = FeaturePipeline()
    res = pipeline.transform(df)

    # Verify key expected features exist
    assert "dV_dt" in res.columns
    assert "dT_dt" in res.columns
    assert "temp_dev_ambient" in res.columns
    assert "dod" in res.columns
    assert "inter_V_x_I" in res.columns
    assert "effective_resistance_proxy" in res.columns
    assert "equivalent_full_cycles" in res.columns

    # Verify no nulls remain in engineered features
    assert res["effective_resistance_proxy"].isna().sum() == 0
    assert res["dV_dt"].isna().sum() == 0
    assert res["dT_dt"].isna().sum() == 0


def test_causality_feature_invariance():
    """Modifying future steps must not alter past feature values."""
    n = 50
    df1 = pd.DataFrame({
        "battery_id": ["BAT_TEST"] * n,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10s"),
        "cycle_number": [1] * n,
        "voltage": np.linspace(4.1, 3.2, n),
        "current": [10.0] * n,
        "temperature": np.linspace(25.0, 30.0, n),
        "power": np.linspace(41.0, 32.0, n),
        "soc": np.linspace(90.0, 30.0, n),
    })

    df2 = df1.copy()
    # Modify future rows starting from index 30
    df2.loc[30:, "voltage"] = 2.5
    df2.loc[30:, "temperature"] = 55.0

    pipeline = FeaturePipeline()
    res1 = pipeline.transform(df1)
    res2 = pipeline.transform(df2)

    # Past rows 0 to 29 must be EXACTLY identical
    pd.testing.assert_series_equal(
        res1.loc[:29, "dV_dt"], res2.loc[:29, "dV_dt"]
    )
    pd.testing.assert_series_equal(
        res1.loc[:29, "temp_dev_ambient"], res2.loc[:29, "temp_dev_ambient"]
    )
    pd.testing.assert_series_equal(
        res1.loc[:29, "effective_resistance_proxy"], res2.loc[:29, "effective_resistance_proxy"]
    )
