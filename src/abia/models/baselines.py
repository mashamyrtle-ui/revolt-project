"""
Machine Learning Baseline Models for SOH Estimation.
Implements Linear Regression, Ridge, Random Forest, and Gradient Boosting.
Computes MAE, RMSE, R2, Max Error, and MAPE across Train, Validation, and Test sets.
Flags empirical overfitting if training error << test error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    # MAPE: avoid division by zero
    non_zero = np.abs(y_true) > 1e-4
    if non_zero.any():
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Max_Error": round(max_err, 4),
        "MAPE": round(mape, 4),
    }


class BaselineModelSuite:
    """Manages baseline regression models for battery SOH prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Linear_Regression": LinearRegression(),
            "Ridge_Regression": Ridge(alpha=1.0, random_state=random_state),
            "Random_Forest": RandomForestRegressor(
                n_estimators=50, max_depth=10, min_samples_split=5, random_state=random_state, n_jobs=1
            ),
            "Gradient_Boosting": HistGradientBoostingRegressor(
                max_iter=100, max_depth=8, l2_regularization=0.1, random_state=random_state
            ),
        }
        self.fitted_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, Any]]:
        """Fits all baseline models and evaluates on train, val, and test splits."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models[name] = model

            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            train_metrics = compute_metrics(y_train, train_preds)
            val_metrics = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            # Check for overfitting: train R2 significantly higher than test R2 or train MAE < 0.2 * test MAE
            overfitting_flag = False
            overfitting_reason = "Good generalization"
            if train_metrics["R2"] > 0.99 and test_metrics["R2"] < 0.85:
                overfitting_flag = True
                overfitting_reason = "Severe overfitting: Train R2 > 0.99 but Test R2 degraded"
            elif train_metrics["MAE"] < 0.3 * test_metrics["MAE"]:
                overfitting_flag = True
                overfitting_reason = "Moderate overfitting: Train MAE significantly lower than Test MAE"

            self.results[name] = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "overfitting_detected": overfitting_flag,
                "generalization_diagnosis": overfitting_reason,
            }

        return self.results
