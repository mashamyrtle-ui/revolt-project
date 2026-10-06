"""
Tests for Phase 1: Dataset Inspection & Schema Mapping.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.abia.data.inspector import DatasetInspector


def test_inspector_detects_canonical_columns():
    inspector = DatasetInspector()
    df = pd.DataFrame({
        "cell_voltage": [3.7, 3.8, 3.9],
        "load_current": [10.0, 12.0, 15.0],
        "cell_temp": [25.0, 26.0, 27.0],
        "state_of_charge": [80.0, 85.0, 90.0],
        "health_pct": [95.0, 95.0, 95.0],
        "timestamp": ["2025-01-01 10:00:00", "2025-01-01 10:00:10", "2025-01-01 10:00:20"],
        "device_id": ["MOD_A", "MOD_A", "MOD_A"],
    })

    report = inspector.inspect_dataframe(df, file_name="test_dataset")
    mappings = report["canonical_mappings"]

    assert mappings["voltage"] == "cell_voltage"
    assert mappings["current"] == "load_current"
    assert mappings["temperature"] == "cell_temp"
    assert mappings["soc"] == "state_of_charge"
    assert mappings["soh"] == "health_pct"
    assert mappings["battery_id"] == "device_id"
    assert report["summary"]["total_rows"] == 3
    assert report["summary"]["number_of_batteries"] == 1


def test_inspector_marks_missing_features_as_unavailable():
    inspector = DatasetInspector()
    # Dataframe with voltage and current, but no temperature or time
    df = pd.DataFrame({
        "voltage": [3.6, 3.7],
        "current": [5.0, 6.0],
    })

    report = inspector.inspect_dataframe(df)
    fa = report["feature_availability"]

    assert fa["electrical_features"]["supported"] is True
    assert fa["thermal_features"]["supported"] is False
    assert fa["thermal_features"]["status"] == "Unavailable"
    assert "temperature" in fa["thermal_features"]["missing_inputs"]
    assert fa["temporal_features"]["supported"] is False
