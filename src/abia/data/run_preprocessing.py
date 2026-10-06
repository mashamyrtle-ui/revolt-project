"""
Runs preprocessing pipeline on raw data and outputs to data/processed/.
"""

import sys
from pathlib import Path
from src.abia.data.preprocessor import DataPreprocessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    raw_path = Path("data/raw/battery_telemetry_benchmark.csv")
    if not raw_path.exists():
        raw_files = list(Path("data/raw").glob("*.csv"))
        if raw_files:
            raw_path = raw_files[0]
        else:
            print("No raw file found.")
            return

    preprocessor = DataPreprocessor()
    print(f"Preprocessing raw dataset: {raw_path}...")
    cleaned_df = preprocessor.preprocess(
        raw_path, output_path="data/processed/battery_telemetry_cleaned.csv"
    )
    print("Preprocessing completed:")
    for k, v in preprocessor.preprocessing_log.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
