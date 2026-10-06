"""
FastAPI Server for ABIA Interactive Battery Intelligence Dashboard.
Serves telemetry API endpoints, benchmarks, reports, and modern frontend assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ABIA Battery Intelligence Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = Path("reports")
STATIC_DIR = Path("src/abia/dashboard/static")
DATA_FILE = Path("data/processed/battery_telemetry_cleaned.csv")


@app.get("/api/batteries")
def get_batteries() -> List[Dict[str, Any]]:
    """Returns list of all available batteries and their high-level circularity status."""
    report_files = list(REPORTS_DIR.glob("ABIA_Intelligence_Report_*.json"))
    summaries = []
    for rf in report_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append({
                    "battery_id": data["battery_id"],
                    "current_soh": data["current_soh"],
                    "future_soh": data["future_soh"],
                    "abhi": data["abhi"],
                    "health_momentum": data["health_momentum"],
                    "anomaly_risk": data["anomaly_risk"],
                    "confidence_pct": data["confidence_pct"],
                    "decision": data["final_circularity_decision"],
                    "css": data["circularity_suitability_score"],
                    "safety_override": data["safety_override_triggered"],
                })
        except Exception:
            continue

    # Sort by battery ID
    return sorted(summaries, key=lambda x: x["battery_id"])


@app.get("/api/battery/{battery_id}")
def get_battery_detail(battery_id: str) -> Dict[str, Any]:
    """Returns full intelligence report and downsampled telemetry timeline for charts."""
    report_file = REPORTS_DIR / f"ABIA_Intelligence_Report_{battery_id}.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail=f"Report for battery {battery_id} not found.")

    with open(report_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Load telemetry timeline for charts
    timeline = []
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        b_df = df[df["battery_id"] == battery_id]
        if len(b_df) > 0:
            # Downsample to ~120 points for smooth charting
            step = max(1, len(b_df) // 120)
            sampled = b_df.iloc[::step].copy()
            for _, r in sampled.iterrows():
                timeline.append({
                    "timestamp": str(r.get("timestamp", "")),
                    "cycle": int(r.get("cycle_number", 0)),
                    "voltage": float(r.get("voltage", 0.0)),
                    "current": float(r.get("current", 0.0)),
                    "temperature": float(r.get("temperature", 0.0)),
                    "soh": float(r.get("soh", 0.0)),
                    "is_anomaly": int(r.get("is_anomaly", 0)),
                })

    report_data["telemetry_timeline"] = timeline
    return report_data


@app.get("/api/benchmarks")
def get_benchmarks() -> Dict[str, Any]:
    """Returns baseline models vs ABIA predictor evaluation metrics."""
    bench_file = REPORTS_DIR / "model_evaluation_benchmarks.json"
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Benchmarks file not found.")

    with open(bench_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Mount static frontend
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ABIA Dashboard API is running. Static UI not yet deployed."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.abia.dashboard.app:app", host="127.0.0.1", port=8000, reload=False)
