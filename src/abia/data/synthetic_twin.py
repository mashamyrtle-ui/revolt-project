"""
Realistic Battery Digital Twin Dataset Generator for Li-ion EV Modules.
Generates multi-cell cycling data adhering to electro-thermal dynamics and degradation kinetics.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def ocv_from_soc(soc_pct: np.ndarray) -> np.ndarray:
    """Non-linear Open Circuit Voltage (OCV) curve for NMC/graphite Li-ion cell."""
    s = np.clip(soc_pct / 100.0, 0.001, 0.999)
    # Empirical 6th order polynomial fit for realistic Li-ion OCV
    ocv = (
        3.14
        + 0.95 * s
        - 0.40 * (s**2)
        + 0.85 * (s**3)
        - 0.55 * (s**4)
        + 0.22 * np.log(s + 1e-4)
        - 0.05 * np.log(1.001 - s)
    )
    return np.clip(ocv, 2.80, 4.25)


def generate_battery_benchmark_dataset(
    output_path: Optional[str | Path] = "data/raw/battery_telemetry_benchmark.csv",
    num_cycles: int = 120,
    samples_per_cycle: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic multi-battery dataset for validating ABIA.
    Profiles represent distinct operational histories:
      - BAT_MOD_01: Nominal gentle cycling (healthy EV battery)
      - BAT_MOD_02: Fast charging stress (moderate degradation)
      - BAT_MOD_03: High ambient temperature aging (second-life candidate)
      - BAT_MOD_04: High mileage, deep discharges (near end-of-life recycle)
      - BAT_MOD_05: Defective module exhibiting thermal anomalies / internal shorts
    """
    np.random.seed(random_seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    nominal_capacity_ah = 50.0  # 50 Ah automotive module
    records: List[Dict] = []
    base_time = datetime.datetime(2025, 1, 1, 8, 0, 0)
    current_time = base_time

    profiles = [
        {"id": "BAT_MOD_01", "deg_rate": 0.00035, "temp_bias": 0.0, "r_init": 0.015, "r_growth": 0.00004, "anomaly": False},
        {"id": "BAT_MOD_02", "deg_rate": 0.00105, "temp_bias": 2.0, "r_init": 0.016, "r_growth": 0.00008, "anomaly": False},
        {"id": "BAT_MOD_03", "deg_rate": 0.00155, "temp_bias": 3.0, "r_init": 0.018, "r_growth": 0.00010, "anomaly": False},
        {"id": "BAT_MOD_04", "deg_rate": 0.00280, "temp_bias": 3.0, "r_init": 0.024, "r_growth": 0.00025, "anomaly": False},
        {"id": "BAT_MOD_05", "deg_rate": 0.00120, "temp_bias": 4.0, "r_init": 0.020, "r_growth": 0.00012, "anomaly": True},
    ]

    for p in profiles:
        bat_id = p["id"]
        deg_rate = p["deg_rate"]
        temp_bias = p["temp_bias"]
        r_init = p["r_init"]
        r_growth = p["r_growth"]
        has_anomaly = p["anomaly"]

        cur_capacity = nominal_capacity_ah
        t_amb = 25.0 + temp_bias
        cell_temp = t_amb

        for cycle in range(1, num_cycles + 1):
            # SEI layer growth and capacity loss
            # Accelerating degradation knee after cycle 80
            acc_factor = 1.0 + (max(0, cycle - 75) ** 1.3) * 0.025
            loss = (deg_rate * (cycle ** 0.55)) * acc_factor + np.random.normal(0, 0.005)
            cur_capacity = max(nominal_capacity_ah * 0.45, nominal_capacity_ah * (1.0 - loss))
            soh = (cur_capacity / nominal_capacity_ah) * 100.0

            # Internal resistance proxy growth
            r_eff = r_init + (r_growth * cycle) * acc_factor + np.random.normal(0, 0.0003)

            # Cycle through discharge-rest-charge-rest
            soc_profile = np.linspace(95.0, 20.0, samples_per_cycle)
            ocv_profile = ocv_from_soc(soc_profile)

            for step in range(samples_per_cycle):
                soc = soc_profile[step]
                ocv = ocv_profile[step]

                # Dynamic current demands (Discharge pulses and regeneration)
                base_current = 25.0 + 10.0 * np.sin(step * 0.6) + np.random.normal(0, 3.0)
                # Occasional regenerative braking current pulse (negative)
                if step % 7 == 0:
                    current_val = -15.0 + np.random.normal(0, 2.0)
                else:
                    current_val = base_current

                # Terminal Voltage: V_term = OCV - I * R_eff + noise
                terminal_voltage = ocv - (current_val * r_eff) + np.random.normal(0, 0.005)
                terminal_voltage = float(np.clip(terminal_voltage, 2.4, 4.35))

                # Thermal dynamics: m*Cp*dT/dt = I^2 * R - h*A*(T - T_amb)
                joule_heat = (current_val**2) * r_eff * 0.04
                cooling = 0.08 * (cell_temp - t_amb)
                cell_temp += joule_heat - cooling + np.random.normal(0, 0.1)

                is_anomaly_step = False
                # Inject realistic anomaly for BAT_MOD_05 in later cycles
                if has_anomaly and cycle >= 95 and step >= 20:
                    # Thermal excursion simulation (local short or cooling failure)
                    cell_temp += 3.5 * (step - 19)
                    terminal_voltage -= 0.35  # sudden voltage collapse
                    is_anomaly_step = True

                power_val = terminal_voltage * current_val

                records.append({
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "battery_id": bat_id,
                    "cycle_number": cycle,
                    "cycle_step": step + 1,
                    "voltage": round(terminal_voltage, 4),
                    "current": round(current_val, 3),
                    "temperature": round(float(cell_temp), 2),
                    "power": round(power_val, 3),
                    "soc": round(float(soc), 2),
                    "capacity_ah": round(float(cur_capacity), 4),
                    "soh": round(float(soh), 3),
                    "is_anomaly": int(is_anomaly_step),
                })

                current_time += datetime.timedelta(seconds=10)

            # Rest period cooling between cycles to ambient
            cell_temp = t_amb + np.random.normal(0, 0.1)

    df = pd.DataFrame(records)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Generated benchmark dataset: {output_path} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    generate_battery_benchmark_dataset()
