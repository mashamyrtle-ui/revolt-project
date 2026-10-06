"""
Aetherix Battery Health Index (ABHI) Engine.
Computes and compares:
  1. Equal weights
  2. Learned fixed weights (constrained optimization on training data)
  3. Context-aware dynamic weights
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ABHIEngine:
    """Computes ABHI with learned, constrained, or context-aware weighting."""

    INDEX_NAMES = ["VSI", "TSI", "CSI", "PSI", "DGI"]

    def __init__(self, min_weight: float = 0.05):
        self.min_weight = min_weight
        self.learned_weights: Optional[np.ndarray] = None
        self.equal_weights: np.ndarray = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        self.comparison_results: Dict[str, Dict[str, float]] = {}

    def fit_weights(
        self,
        indices_df: pd.DataFrame,
        target_health_series: pd.Series,
    ) -> np.ndarray:
        """
        Learns optimal fixed weights on TRAINING DATA ONLY using constrained optimization.
        Objective: Minimize MSE between ABHI and target health proxy (e.g., SOH),
        subject to: sum(w_i) = 1.0 and w_i >= min_weight.
        """
        X = indices_df[self.INDEX_NAMES].values
        y = target_health_series.values

        # Remove any invalid pairs
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        def loss_fn(w: np.ndarray) -> float:
            pred = np.dot(X_clean, w)
            return float(np.mean((pred - y_clean) ** 2))

        init_w = np.ones(len(self.INDEX_NAMES)) / len(self.INDEX_NAMES)
        bounds = [(self.min_weight, 0.70) for _ in range(len(self.INDEX_NAMES))]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            loss_fn,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success:
            self.learned_weights = res.x / np.sum(res.x)
        else:
            self.learned_weights = self.equal_weights.copy()

        return self.learned_weights

    def compute_abhi(
        self,
        indices_df: pd.DataFrame,
        method: str = "learned_constrained",
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Computes ABHI for each row according to the specified weighting method.
        """
        X = indices_df[self.INDEX_NAMES].values

        if method == "equal_weights":
            w = self.equal_weights
            abhi = np.dot(X, w)

        elif method == "learned_constrained":
            w = self.learned_weights if self.learned_weights is not None else self.equal_weights
            abhi = np.dot(X, w)

        elif method == "context_aware":
            # Dynamic weighting: when temperature is high, amplify TSI weight
            # When current is severe, amplify CSI weight
            weights_matrix = np.tile(
                self.learned_weights if self.learned_weights is not None else self.equal_weights,
                (len(indices_df), 1),
            )
            if context_df is not None and "temperature" in context_df.columns:
                t = context_df["temperature"].values
                # Thermal stress multiplier if T > 35 C
                t_stress = np.maximum(0.0, (t - 35.0) / 15.0)
                weights_matrix[:, 1] += t_stress * 0.15  # TSI weight boost
                # Normalize each row to sum to 1
                row_sums = weights_matrix.sum(axis=1, keepdims=True)
                weights_matrix = weights_matrix / row_sums

            abhi = np.sum(X * weights_matrix, axis=1)

        else:
            raise ValueError(f"Unknown ABHI weighting method: {method}")

        return pd.Series(np.clip(abhi, 0.0, 100.0), index=indices_df.index, name="ABHI")

    def compare_weighting_formulations(
        self,
        indices_df: pd.DataFrame,
        target_health_series: pd.Series,
        context_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compares equal, learned, and context-aware formulations on validation data.
        Reports MAE, RMSE, and correlation with target health.
        """
        results = {}
        methods = ["equal_weights", "learned_constrained", "context_aware"]

        y = target_health_series.values
        for m in methods:
            pred = self.compute_abhi(indices_df, method=m, context_df=context_df).values
            mae = float(np.mean(np.abs(pred - y)))
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else 0.0

            results[m] = {
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "Pearson_Correlation": round(corr, 4),
            }

        self.comparison_results = results
        return results
