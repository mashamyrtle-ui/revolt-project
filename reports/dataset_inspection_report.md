# ABIA Dataset Automated Inspection Report

## 1. Executive Summary
- **File Name:** `battery_telemetry_benchmark.csv`
- **File Size:** 1520.50 KB
- **Total Samples:** 18,000
- **Total Columns:** 12
- **Duplicate Rows:** 0
- **Batteries / Modules Identified:** 5
- **Battery IDs:** BAT_MOD_01, BAT_MOD_02, BAT_MOD_03, BAT_MOD_04, BAT_MOD_05

## 2. Inferred Canonical Semantics
| Physical Semantic | Mapped Column Name | Status |
| :--- | :--- | :--- |
| Voltage | `voltage` | Found |
| Current | `current` | Found |
| Temperature | `temperature` | Found |
| Power | `power` | Found |
| Soc | `soc` | Found |
| Soh | `soh` | Found |
| Cycle | `cycle_number` | Found |
| Capacity | `capacity_ah` | Found |
| Timestamp | `timestamp` | Found |
| Battery_id | `battery_id` | Found |
| Diagnostic | `is_anomaly` | Found |

## 3. Chronological Coverage
- **Timestamp Available:** True
- **Coverage Interval:** 2025-01-01 08:00:00 to 2025-01-03 09:59:50
- **Median Δt:** 10.0 seconds
- **Estimated Sampling Frequency:** 0.1 Hz
- **Strictly Monotonic Time:** True

## 4. Feature Availability Matrix
| Feature Category | Support Status | Missing Prerequisites | Notes |
| :--- | :--- | :--- | :--- |
| Electrical Features | Available | None | Standard |
| Thermal Features | Available | None | Standard |
| Temporal Features | Available | None | Standard |
| Usage Soc Features | Available | None | Standard |
| Effective Resistance Proxy | Available (Proxy dV/dI) | None | Named effective resistance proxy, not true internal resistance. |
| Coulomb Counting Capacity | Available | None | Standard |
| Equivalent Full Cycles | Available | None | Standard |
| Target Current Soh | Available | None | Standard |

## 5. Column Diagnostics & Quality
| Column | Dtype | Nulls (%) | Min | Max | Mean | Std | Infinite |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `timestamp` | str | 0.0% | - | - | - | - | 0 |
| `battery_id` | str | 0.0% | - | - | - | - | 0 |
| `cycle_number` | int64 | 0.0% | 1.000 | 120.000 | 60.500 | 34.641 | 0 |
| `cycle_step` | int64 | 0.0% | 1.000 | 30.000 | 15.500 | 8.656 | 0 |
| `voltage` | float64 | 0.0% | 2.050 | 4.350 | 2.960 | 0.607 | 0 |
| `current` | float64 | 0.0% | -21.751 | 44.466 | 18.945 | 16.778 | 0 |
| `temperature` | float64 | 0.0% | 24.950 | 259.680 | 46.508 | 27.044 | 0 |
| `power` | float64 | 0.0% | -94.618 | 142.296 | 47.990 | 52.475 | 0 |
| `soc` | float64 | 0.0% | 20.000 | 95.000 | 57.500 | 22.385 | 0 |
| `capacity_ah` | float64 | 0.0% | 41.078 | 50.471 | 48.547 | 1.645 | 0 |
| `soh` | float64 | 0.0% | 82.156 | 100.942 | 97.093 | 3.290 | 0 |
| `is_anomaly` | int64 | 0.0% | 0.000 | 1.000 | 0.014 | 0.119 | 0 |