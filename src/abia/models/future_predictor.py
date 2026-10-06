"""
Future SOH Prediction Engine for ABIA.
Predicts battery health at configurable forward horizons (short, medium, long term)
using strictly causal historical features at prediction time t.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.abia.models.baselines import compute_metrics


class FutureSOHPredictor:
    """Multi-horizon forecasting engine for battery SOH."""

    def __init__(
        self,
        horizons: Optional[Dict[str, int]] = None,
        random_state: int = 42,
    ):
        # Default step horizons (e.g., 30 steps ≈ 1 cycle, 150 steps ≈ 5 cycles, etc.)
        self.horizons = horizons or {"short_term": 30, "medium_term": 150, "long_term": 300}
        self.random_state = random_state
        self.models: Dict[str, HistGradientBoostingRegressor] = {
            h_name: HistGradientBoostingRegressor(
                max_iter=70, max_depth=5, l2_regularization=3.0, random_state=random_state
            )
            for h_name in self.horizons
        }
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False
        self.evaluation_results: Dict[str, Dict[str, float]] = {}

    def _create_horizon_targets(
        self,
        df: pd.DataFrame,
        soh_col: str = "soh",
        battery_col: str = "battery_id",
    ) -> Dict[str, pd.Series]:
        """Creates forward-shifted target series per battery without cross-battery contamination."""
        targets: Dict[str, pd.Series] = {}
        for h_name, h_steps in self.horizons.items():
            shifted_list = []
            for _, grp in df.groupby(battery_col, sort=False):
                # Shift target backwards so row t has SOH[t + h_steps]
                shifted = grp[soh_col].shift(-h_steps)
                shifted_list.append(shifted)
            targets[h_name] = pd.concat(shifted_list).reindex(df.index)
        return targets

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        soh_col: str = "soh",
        battery_col: str = "battery_id",
    ) -> "FutureSOHPredictor":
        """Fits multi-horizon forecasting models on training timeline only."""
        self.feature_columns = feature_cols
        X_train = train_df[feature_cols].copy().ffill().fillna(0.0)

        targets = self._create_horizon_targets(train_df, soh_col=soh_col, battery_col=battery_col)

        for h_name, y_target in targets.items():
            valid_mask = ~y_target.isna()
            X_valid = X_train[valid_mask]
            y_valid = y_target[valid_mask].values

            if len(X_valid) > 20:
                self.models[h_name].fit(X_valid, y_valid)

        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon_name: str = "short_term",
    ) -> np.ndarray:
        """Forecasts future SOH for the specified horizon."""
        if not self.is_fitted:
            raise RuntimeError("FutureSOHPredictor must be fitted before predict.")

        if horizon_name not in self.models:
            raise ValueError(f"Unknown horizon {horizon_name}. Valid: {list(self.models.keys())}")

        X = df[self.feature_columns].copy().ffill().fillna(0.0)
        preds = self.models[horizon_name].predict(X)
        return np.clip(preds, 0.0, 105.0)

    def evaluate_on_unseen(
        self,
        test_df: pd.DataFrame,
        soh_col: str = "soh",
        battery_col: str = "battery_id",
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates future SOH forecasting on unseen chronological test timeline."""
        targets = self._create_horizon_targets(test_df, soh_col=soh_col, battery_col=battery_col)
        results = {}

        for h_name in self.horizons:
            y_true_series = targets[h_name]
            valid_mask = ~y_true_series.isna()
            if valid_mask.sum() < 5:
                continue

            test_valid = test_df[valid_mask]
            y_true = y_true_series[valid_mask].values
            y_pred = self.predict(test_valid, horizon_name=h_name)

            metrics = compute_metrics(y_true, y_pred)
            results[h_name] = metrics

        self.evaluation_results = results
        return results
