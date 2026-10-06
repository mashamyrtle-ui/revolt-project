"""
Grounded Confidence Engine for ABIA.
Quantifies epistemic and distributional confidence in SOH predictions.
Decreases confidence when samples are out-of-distribution, model divergence is high,
or operating conditions fall outside familiar training bounds.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ConfidenceEngine:
    """Computes mathematically grounded prediction confidence percentages (0-100%)."""

    def __init__(self):
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.temp_min_99: float = 10.0
        self.temp_max_99: float = 45.0
        self.curr_max_99: float = 50.0
        self.feature_names: List[str] = []
        self.is_calibrated: bool = False

    def calibrate(self, X_train: pd.DataFrame) -> "ConfidenceEngine":
        """Calibrates baseline distributional statistics from training dataset."""
        num_df = X_train.select_dtypes(include=[np.number])
        self.feature_names = list(num_df.columns)
        self.feature_means = num_df.mean().values
        self.feature_stds = num_df.std().replace(0.0, 1.0).values

        if "temperature" in X_train.columns:
            self.temp_min_99 = float(X_train["temperature"].quantile(0.01))
            self.temp_max_99 = float(X_train["temperature"].quantile(0.99))

        if "current" in X_train.columns:
            self.curr_max_99 = float(X_train["current"].abs().quantile(0.99))

        self.is_calibrated = True
        return self

    def compute_confidence(
        self,
        X_sample: pd.Series | pd.DataFrame,
        anchor_soh: Optional[float] = None,
        predicted_soh: Optional[float] = None,
    ) -> float:
        """
        Calculates grounded prediction confidence percentage for a single sample.
        """
        if not self.is_calibrated or self.feature_means is None:
            return 85.0

        if isinstance(X_sample, pd.DataFrame):
            sample_series = X_sample.iloc[-1]
        else:
            sample_series = X_sample

        # 1. Distributional Distance (Z-score magnitude across shared features)
        common_cols = [c for c in self.feature_names if c in sample_series.index]
        if common_cols:
            indices = [self.feature_names.index(c) for c in common_cols]
            means = self.feature_means[indices]
            stds = self.feature_stds[indices]
            vals = sample_series[common_cols].values.astype(float)
            z_scores = (vals - means) / stds
            # Normalized Euclidean distance
            dist_score = float(np.sqrt(np.mean(z_scores**2)) / 3.0)
            dist_score = np.clip(dist_score, 0.0, 1.0)
        else:
            dist_score = 0.2

        # 2. Model Divergence (Disagreement between anchor and boosted prediction)
        if anchor_soh is not None and predicted_soh is not None:
            disagreement = abs(anchor_soh - predicted_soh)
            div_score = float(np.clip(disagreement / 8.0, 0.0, 1.0))
        else:
            div_score = 0.1

        # 3. Operational Familiarity Penalty (Temperature & Current bounds)
        op_penalty = 0.0
        temp = sample_series.get("temperature")
        if temp is not None:
            if temp > self.temp_max_99:
                op_penalty += 0.3 * min(1.0, (temp - self.temp_max_99) / 10.0)
            elif temp < self.temp_min_99:
                op_penalty += 0.3 * min(1.0, (self.temp_min_99 - temp) / 10.0)

        curr = sample_series.get("current")
        if curr is not None and abs(curr) > self.curr_max_99:
            op_penalty += 0.2 * min(1.0, (abs(curr) - self.curr_max_99) / 20.0)

        total_uncertainty = 0.40 * dist_score + 0.35 * div_score + 0.25 * op_penalty
        confidence = 100.0 * (1.0 - np.clip(total_uncertainty, 0.0, 0.85))

        return round(float(confidence), 1)
