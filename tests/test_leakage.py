"""
Tests for Phase 5: Chronological Splitting & Leakage Prevention.
"""

import pandas as pd
import pytest
from src.abia.data.splitter import TimeSeriesSplitter


def test_chronological_split_preserves_temporal_order():
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    n = 100
    df = pd.DataFrame({
        "battery_id": ["BAT_01"] * n,
        "cycle_number": list(range(1, n + 1)),
        "voltage": [3.7] * n,
    })

    train_df, val_df, test_df = splitter.chronological_split(
        df, battery_col="battery_id", time_or_cycle_col="cycle_number"
    )

    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20

    assert train_df["cycle_number"].max() == 60
    assert val_df["cycle_number"].min() == 61
    assert val_df["cycle_number"].max() == 80
    assert test_df["cycle_number"].min() == 81
    assert test_df["cycle_number"].max() == 100

    # Formal leakage verification check
    leakage_report = splitter.verify_no_temporal_leakage(
        train_df, val_df, test_df, time_or_cycle_col="cycle_number"
    )
    assert leakage_report["passed"] is True
    assert leakage_report["leakage_detected"] is False
