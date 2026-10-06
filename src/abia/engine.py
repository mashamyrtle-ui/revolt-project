"""
ABIA Core Intelligence Engine & Pipeline Orchestrator.
Coordinates data ingestion, feature generation, health index calculation,
ABHI weight optimization, model training & evaluation, risk diagnosis,
safety overrides, and circularity decisioning.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

from src.abia.data.splitter import TimeSeriesSplitter
from src.abia.features.pipeline import FeaturePipeline
from src.abia.indices.health_indices import compute_all_health_indices
from src.abia.indices.abhi import ABHIEngine
from src.abia.dynamics.momentum import HealthMomentumEngine
from src.abia.dynamics.fingerprint import BatteryBehaviourFingerprint
from src.abia.models.baselines import BaselineModelSuite
from src.abia.models.abia_predictor import ABIAPredictor
from src.abia.models.future_predictor import FutureSOHPredictor
from src.abia.intelligence.safety import SafetyEngine
from src.abia.intelligence.confidence import ConfidenceEngine
from src.abia.intelligence.anomaly import AnomalyRiskEngine
from src.abia.intelligence.circularity import CircularityEngine
from src.abia.report.generator import BatteryIntelligenceReportGenerator


class ABIAEngine:
    """Master orchestrator for the ABIA battery intelligence framework."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_pipeline = FeaturePipeline()
        self.abhi_engine = ABHIEngine(min_weight=0.05)
        self.momentum_engine = HealthMomentumEngine()
        self.fingerprint_engine = BatteryBehaviourFingerprint()
        self.baseline_suite = BaselineModelSuite(random_state=random_state)
        self.abia_predictor = ABIAPredictor(random_state=random_state)
        self.future_predictor = FutureSOHPredictor(random_state=random_state)
        self.safety_engine = SafetyEngine()
        self.confidence_engine = ConfidenceEngine()
        self.anomaly_engine = AnomalyRiskEngine()
        self.circularity_engine = CircularityEngine()

        self.enriched_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.benchmark_results: Dict[str, Any] = {}
        self.battery_reports: Dict[str, Dict[str, Any]] = {}

    def run_full_pipeline(
        self,
        data_path: str = "data/processed/battery_telemetry_cleaned.csv",
        output_dir: str = "reports",
    ) -> Dict[str, Any]:
        """Executes full end-to-end ABIA pipeline and evaluation."""
        print("=" * 60)
        print("STARTING ABIA BATTERY INTELLIGENCE PIPELINE")
        print("=" * 60)

        # 1. Load Data
        df = pd.read_csv(data_path)
        print(f"[1/7] Ingested {len(df):,} cleaned telemetry records across {df['battery_id'].nunique()} batteries.")

        # 2. Feature Engineering
        print("[2/7] Computing causal features (Electrical, Thermal, Usage, Temporal, Interactions, Physics)...")
        feats_df = self.feature_pipeline.transform(df)

        # 3. Health Indices & Momentum
        print("[3/7] Calculating health indices (VSI, TSI, CSI, PSI, DGI) & Health Momentum...")
        indices_df = compute_all_health_indices(feats_df)
        momentum_df = self.momentum_engine.compute_momentum(feats_df)
        self.enriched_df = pd.concat([feats_df, indices_df, momentum_df], axis=1)
        self.enriched_df = self.enriched_df.loc[:, ~self.enriched_df.columns.duplicated()]

        # 4. Chronological Splitting (Leakage-Proof)
        print("[4/7] Performing chronological split (60% Train, 20% Val, 20% Test per battery)...")
        splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        self.train_df, self.val_df, self.test_df = splitter.chronological_split(
            self.enriched_df, battery_col="battery_id", time_or_cycle_col="cycle_number"
        )

        leakage_check = splitter.verify_no_temporal_leakage(
            self.train_df, self.val_df, self.test_df, time_or_cycle_col="cycle_number"
        )
        print(f"      Chronological Barrier Verification: Passed={leakage_check['passed']}")

        # 5. Fit ABHI Weight Learning on Training Data
        print("[5/7] Fitting constrained ABHI weight learning on training data...")
        self.abhi_engine.fit_weights(self.train_df, self.train_df["soh"])
        abhi_comparison = self.abhi_engine.compare_weighting_formulations(
            self.val_df, self.val_df["soh"], context_df=self.val_df
        )
        print(f"      Learned Fixed Weights: {dict(zip(self.abhi_engine.INDEX_NAMES, np.round(self.abhi_engine.learned_weights, 3)))}")
        print(f"      ABHI Formulations Comparison on Validation: {abhi_comparison}")

        # Compute ABHI column across all splits
        self.train_df["ABHI"] = self.abhi_engine.compute_abhi(self.train_df, method="learned_constrained")
        self.val_df["ABHI"] = self.abhi_engine.compute_abhi(self.val_df, method="learned_constrained")
        self.test_df["ABHI"] = self.abhi_engine.compute_abhi(self.test_df, method="learned_constrained")
        self.enriched_df["ABHI"] = self.abhi_engine.compute_abhi(self.enriched_df, method="learned_constrained")

        # Calibrate Confidence Engine on training features
        self.confidence_engine.calibrate(self.train_df)

        # 6. Model Training & Evaluation (Baselines vs ABIA)
        print("[6/7] Training & benchmarking baseline ML models vs ABIA Predictor...")
        feature_cols = [
            "VSI", "TSI", "CSI", "PSI", "DGI", "ABHI",
            "effective_resistance_proxy", "resistance_growth_ratio",
            "equivalent_full_cycles", "dV_dt", "dT_dt", "temp_dev_ambient",
            "thermal_stress_composite", "dod", "inter_V_x_I", "inter_I_x_T",
            "bhm_slope", "bhm_volatility",
        ]
        feature_cols = [c for c in feature_cols if c in self.train_df.columns]

        X_tr = self.train_df[feature_cols].copy().ffill().fillna(0.0)
        y_tr = self.train_df["soh"]
        X_val = self.val_df[feature_cols].copy().ffill().fillna(0.0)
        y_val = self.val_df["soh"]
        X_te = self.test_df[feature_cols].copy().ffill().fillna(0.0)
        y_te = self.test_df["soh"]

        # Train Baselines
        baseline_results = self.baseline_suite.fit_and_evaluate(X_tr, y_tr, X_val, y_val, X_te, y_te)

        # Train ABIA Predictor
        self.abia_predictor.fit(self.train_df, y_tr, self.val_df, y_val)
        abia_test_metrics = self.abia_predictor.evaluate(self.test_df, y_te)

        # Check ABIA overfitting
        abia_overfitting = (
            self.abia_predictor.training_metrics["R2"] > 0.99 and abia_test_metrics["R2"] < 0.85
        )
        self.benchmark_results = {
            "baselines": baseline_results,
            "abia_predictor": {
                "train": self.abia_predictor.training_metrics,
                "val": self.abia_predictor.val_metrics,
                "test": abia_test_metrics,
                "overfitting_detected": abia_overfitting,
                "generalization_diagnosis": "Good generalization (Regularized residual fusion)" if not abia_overfitting else "Overfitting detected",
            },
            "abhi_comparison": abhi_comparison,
        }

        # Train Future SOH Predictor
        print("      Training multi-horizon future SOH forecasting models...")
        self.future_predictor.fit(self.train_df, feature_cols=feature_cols)
        future_metrics = self.future_predictor.evaluate_on_unseen(self.test_df)
        self.benchmark_results["future_soh_forecasting"] = future_metrics

        # 7. Generate Battery Reports & Decisions for All Batteries
        print("[7/7] Generating Battery Behaviour Fingerprints & Circularity Decisions...")
        bat_ids = list(self.enriched_df["battery_id"].unique())

        for b_id in bat_ids:
            b_df = self.enriched_df[self.enriched_df["battery_id"] == b_id].copy()
            latest_row = b_df.iloc[-1]

            # Current & Future SOH predictions
            pred_cur_soh = float(self.abia_predictor.predict(b_df.tail(1))[0])
            pred_fut_soh = float(self.future_predictor.predict(b_df.tail(1), horizon_name="short_term")[0])
            deg_rate = self.abia_predictor.estimate_degradation_rate(b_df)

            # Health Momentum & Fingerprint
            momentum_status = str(latest_row.get("bhm_status", "Stable"))
            fingerprint = self.fingerprint_engine.compute_fingerprint(b_df)

            # Grounded Confidence
            anchor_val = float(latest_row.get("ABHI", pred_cur_soh))
            confidence = self.confidence_engine.compute_confidence(
                latest_row, anchor_soh=anchor_val, predicted_soh=pred_cur_soh
            )

            # Safety & Risk Evaluation
            safety_status = self.safety_engine.evaluate_safety(latest_row)
            residual = float(latest_row.get("soh", pred_cur_soh) - pred_cur_soh)
            risk_info = self.anomaly_engine.evaluate_risk(latest_row, safety_status, model_residual=residual)

            # Health Indices dict
            indices = {
                "VSI": float(latest_row.get("VSI", 80.0)),
                "TSI": float(latest_row.get("TSI", 80.0)),
                "CSI": float(latest_row.get("CSI", 80.0)),
                "PSI": float(latest_row.get("PSI", 80.0)),
                "DGI": float(latest_row.get("DGI", 80.0)),
            }
            abhi_val = float(latest_row.get("ABHI", 80.0))

            # Circularity Decision
            circ_decision = self.circularity_engine.evaluate_decision(
                current_soh=pred_cur_soh,
                future_soh=pred_fut_soh,
                abhi=abhi_val,
                degradation_rate=deg_rate,
                momentum_status=momentum_status,
                vsi=indices["VSI"],
                tsi=indices["TSI"],
                csi=indices["CSI"],
                psi=indices["PSI"],
                dgi=indices["DGI"],
                risk_info=risk_info,
                confidence=confidence,
                safety_status=safety_status,
            )

            rep = BatteryIntelligenceReportGenerator.generate_report(
                battery_id=b_id,
                current_soh=pred_cur_soh,
                future_soh=pred_fut_soh,
                abhi=abhi_val,
                indices=indices,
                health_momentum=momentum_status,
                fingerprint=fingerprint,
                degradation_rate=deg_rate,
                risk_info=risk_info,
                confidence=confidence,
                safety_status=safety_status,
                circularity_decision=circ_decision,
                output_dir=output_dir,
            )
            self.battery_reports[b_id] = rep
            print(f"  -> Battery `{b_id}`: SOH={pred_cur_soh:.1f}% | CSS={rep['circularity_suitability_score']} | Decision: {rep['final_circularity_decision']}")

        # Save Benchmark summary JSON
        bench_path = Path(output_dir) / "model_evaluation_benchmarks.json"
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(self.benchmark_results, f, indent=2)
        print(f"\nAll reports and benchmarks saved to: {output_dir}/")

        return {
            "benchmarks": self.benchmark_results,
            "battery_reports": self.battery_reports,
        }
