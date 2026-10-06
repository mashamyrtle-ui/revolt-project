"""
CLI entry point to execute the complete ABIA pipeline.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.abia.engine import ABIAEngine


def main():
    print("Initializing ABIA pipeline...", flush=True)
    engine = ABIAEngine(random_state=42)
    data_file = "data/processed/battery_telemetry_cleaned.csv"
    if not Path(data_file).exists():
        print(f"Processed file {data_file} not found. Running preprocessor...", flush=True)
        from src.abia.data.preprocessor import DataPreprocessor
        pre = DataPreprocessor()
        pre.preprocess("data/raw/battery_telemetry_benchmark.csv", output_path=data_file)

    results = engine.run_full_pipeline(data_path=data_file, output_dir="reports")
    print("\nABIA Pipeline completed successfully!", flush=True)


if __name__ == "__main__":
    main()
