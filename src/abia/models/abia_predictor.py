"""
ABIA Prediction Engine (Current SOH & Degradation Rate).
A custom, domain-informed battery intelligence framework combining:
  - Physics-proxy anchors (Reff, EFC)
  - Normalized health indices (VSI, TSI, CSI, PSI, DGI)
  - Learned ABHI
  - Temporal dynamics & Health Momentum
  - Residual regularization to prevent overfitting
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from src.abia.models.baselines import compute_metrics


class ABIAPredictor:
    """
    Custom domain-informed battery intelligence framework for SOH estimation.
    Fuses physics-guided degradation anchors with regularized gradient boosting
    to ensure generalization on unseen chronological and module data.
    """

    CORE_FEATURE_NAMES = [
        # Health Indices
        "VSI",
        "TSI",
        "CSI",
        "PSI",
        "DGI",
        "ABHI",
        # Physics Proxies
        "effective_resistance_proxy",
        "resistance_growth_ratio",
        "equivalent_full_cycles",
        # Electrical & Thermal Dynamics
        "dV_dt",
        "dT_dt",
        "temp_dev_ambient",
        "thermal_stress_composite",
        "dod",
        # Interactions
        "inter_V_x_I",
        "inter_I_x_T",
        "response_V_to_I",
        # Health Momentum
        "bhm_slope",
        "bhm_volatility",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Domain anchor model (Ridge regression regularizing towards physics indices)
        self.anchor_model = Ridge(alpha=10.0, random_state=random_state)
        # Residual booster (captures high-order non-linear thermal-electrical interactions)
        self.residual_booster = HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=2.0,  # Strong regularization to prevent overfitting
            random_state=random_state,
        )
        self.selected_features: List[str] = []
        self.is_fitted: bool = False
        self.training_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects available core features and fills any missing causally."""
        available = [f for f in self.CORE_FEATURE_NAMES if f in df.columns]
        if not available:
            # Fallback to numerical features
            num_cols = df.select_dtypes(include=[np.number]).columns
            available = [c for c in num_cols if c not in ["soh", "is_anomaly"]]

        self.selected_features = available
        X = df[available].copy().ffill().fillna(0.0)
        return X

    def fit(
        self,
        X_train_df: pd.DataFrame,
        y_train: pd.Series,
        X_val_df: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "ABIAPredictor":
        """
        Fits ABIA framework:
          1. Fits physics/health anchor on training data
          2. Computes training residuals: r = y - anchor_pred
          3. Fits regularized residual booster on residuals
        """
        X_train = self._prepare_features(X_train_df)
        y_tr = y_train.values

        # 1. Fit anchor model
        self.anchor_model.fit(X_train, y_tr)
        anchor_tr = self.anchor_model.predict(X_train)

        # 2. Residual learning
        residuals_tr = y_tr - anchor_tr
        self.residual_booster.fit(X_train, residuals_tr)

        self.is_fitted = True

        # Calculate train metrics
        tr_preds = self.predict(X_train_df)
        self.training_metrics = compute_metrics(y_tr, tr_preds)

        if X_val_df is not None and y_val is not None:
            val_preds = self.predict(X_val_df)
            self.val_metrics = compute_metrics(y_val.values, val_preds)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts SOH fusing domain anchor with regularized residual.
        Enforces realistic physical clipping to [0.0, 105.0]%.
        """
        if not self.is_fitted:
            raise RuntimeError("ABIAPredictor must be fitted before predict.")

        X = df[self.selected_features].copy().ffill().fillna(0.0)
        anchor_pred = self.anchor_model.predict(X)
        residual_pred = self.residual_booster.predict(X)

        combined_soh = anchor_pred + residual_pred
        return np.clip(combined_soh, 0.0, 105.0)

    def evaluate(
        self,
        X_test_df: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluates on unseen test dataset."""
        preds = self.predict(X_test_df)
        self.test_metrics = compute_metrics(y_test.values, preds)
        return self.test_metrics

    def estimate_degradation_rate(
        self,
        df: pd.DataFrame,
        window: int = 30,
    ) -> float:
        """
        Estimates local degradation rate (% SOH loss per 100 cycles)
        from historical predictions strictly looking backwards.
        """
        if len(df) < 5:
            return 0.05
        preds = self.predict(df.tail(window))
        # Rate of change over window
        delta_soh = preds[-1] - preds[0]
        rate_per_step = -delta_soh / float(len(preds))
        # Scaled to nominal 100 cycles equivalent
        rate_per_100_cycles = max(0.0, rate_per_step * 100.0)
        return round(float(rate_per_100_cycles), 4)
