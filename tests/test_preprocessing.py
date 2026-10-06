"""
Tests for Phase 2: Data Preprocessing and Immutability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.abia.data.preprocessor import DataPreprocessor


def test_preprocessing_deduplication_and_sorting(tmp_path):
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        "battery_id": ["BAT_A", "BAT_A", "BAT_A", "BAT_A"],
        "timestamp": [
            "2025-01-01 10:00:20",
            "2025-01-01 10:00:00",
            "2025-01-01 10:00:10",
            "2025-01-01 10:00:00",  # Duplicate
        ],
        "voltage": [3.8, 3.7, 3.75, 3.7],
        "current": [10.0, 10.0, 10.0, 10.0],
        "temperature": [25.0, 24.5, 24.8, 24.5],
        "soc": [85.0, 90.0, 88.0, 90.0],
    })

    out_file = tmp_path / "processed_test.csv"
    res = preprocessor.preprocess(df, output_path=out_file)

    # Assert duplicate removed: 4 rows -> 3 rows
    assert len(res) == 3
    # Assert chronologically sorted
    times = list(res["timestamp"])
    assert times == [
        "2025-01-01 10:00:00",
        "2025-01-01 10:00:10",
        "2025-01-01 10:00:20",
    ]
    assert out_file.exists()


def test_preprocessing_filters_invalid_physical_readings():
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        "battery_id": ["BAT_A", "BAT_A", "BAT_A"],
        "voltage": [3.7, -1.0, 3.8],  # -1.0 is physically impossible
        "temperature": [25.0, 200.0, 26.0],  # 200 C is impossible
        "soc": [90.0, 150.0, 80.0],  # 150 will be clipped to 100
    })

    res = preprocessor.preprocess(df, output_path=None)
    # The second row (-1.0V and 200C) should be eliminated
    assert len(res) == 2
    assert -1.0 not in res["voltage"].values
