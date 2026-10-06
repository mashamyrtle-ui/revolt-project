"""
CLI script to run automated dataset inspection and generate reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.abia.data.inspector import DatasetInspector


def run_inspection(
    input_file: str = "data/raw/battery_telemetry_benchmark.csv",
    output_dir: str = "reports",
) -> None:
    path = Path(input_file)
    if not path.exists():
        # Check if any csv exists in data/raw
        raw_files = list(Path("data/raw").glob("*.csv")) + list(Path("data/raw").glob("*.parquet"))
        if raw_files:
            path = raw_files[0]
            print(f"Target file not specified, found and inspecting: {path}")
        else:
            print(f"Error: No raw dataset found at {input_file} or in data/raw/")
            sys.exit(1)

    inspector = DatasetInspector()
    print(f"Inspecting dataset: {path}...")
    report = inspector.inspect_file_or_dataframe(path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_inspection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_report = inspector.generate_markdown_report(report)
    md_path = out_dir / "dataset_inspection_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"Inspection report generated successfully:")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    print("\n--- Summary ---")
    print(f"Rows: {report['summary']['total_rows']:,}")
    print(f"Columns: {report['summary']['total_columns']}")
    print(f"Batteries: {report['summary']['number_of_batteries']} ({report['summary']['battery_identifiers']})")
    print("Features Supported:")
    for feat, status in report["feature_availability"].items():
        print(f"  * {feat}: {status['status']}")


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw/battery_telemetry_benchmark.csv"
    run_inspection(file_arg)
