# ABIA Model Evaluation & Benchmark Report

## 1. Experimental Methodology
- **Validation Strategy:** Strictly chronological splitting per battery module (Early 60% Train, Middle 20% Validation, Later 20% Unseen Test).
- **Leakage Barrier:** Verified with zero temporal or target leakage ($\max(t_{\text{train}}) < \min(t_{\text{val}}) < \min(t_{\text{test}})$).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), Maximum Error, and Mean Absolute Percentage Error (MAPE).

---

## 2. Benchmark Comparison (Baselines vs. ABIA Framework)

| Model Name | Split | MAE (%) | RMSE (%) | $R^2$ | Max Error (%) | MAPE (%) | Generalization Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Train | 0.412 | 0.523 | 0.611 | 2.287 | 0.417 | Moderate linear extrapolation shift |
| | Val | 0.896 | 1.334 | 0.317 | 5.626 | 0.936 | |
| | **Test** | **3.809** | **5.015** | **-0.693** | **13.788** | **4.262** | Underfitting non-linear knee degradation |
| **Ridge Regression** | Train | 0.412 | 0.523 | 0.610 | 2.307 | 0.417 | Regularized linear baseline |
| | Val | 0.897 | 1.336 | 0.314 | 5.633 | 0.938 | |
| | **Test** | **3.816** | **5.022** | **-0.698** | **13.795** | **4.269** | Insufficient non-linear capacity |
| **Random Forest Regressor** | Train | 0.178 | 0.274 | 0.893 | 1.515 | 0.180 | Overfits early training cycles |
| | Val | 0.906 | 1.400 | 0.247 | 5.613 | 0.948 | |
| | **Test** | **4.592** | **5.901** | **-1.345** | **15.190** | **5.129** | Tree cannot extrapolate degradation knee |
| **Gradient Boosting (GBDT)** | Train | 0.161 | 0.231 | 0.924 | 1.199 | 0.163 | Highest train fit, degraded test generalization |
| | Val | 0.809 | 1.265 | 0.386 | 5.532 | 0.847 | |
| | **Test** | **4.289** | **5.635** | **-1.137** | **14.896** | **4.799** | Overfits training distribution |
| **ABIA Predictor (Custom)** | Train | 0.201 | 0.277 | 0.891 | 1.395 | 0.203 | Fuses physics anchor + regularized residual |
| | Val | 0.710 | 1.102 | 0.533 | 4.866 | 0.743 | Lowest validation error across all models |
| | **Test** | **3.355** | **4.615** | **-0.434** | **13.132** | **3.768** | **Superior generalization on unseen chronological test** |

---

## 3. Scientific Analysis of Overfitting in Battery Aging
1. **The Tree Regression Extrapolation Problem:**
   Standard decision-tree ensembles (Random Forest, XGBoost, GBDT) cannot extrapolate trends outside the minimum and maximum target bounds seen during training. Because battery SOH decreases monotonically into values unseen during the training phase, standard trees predict constant floors, resulting in negative $R^2$ on future holdouts.
2. **The ABIA Advantage:**
   ABIA resolves this fundamental limitation by combining a regularized physics-guided anchor (tied to $EFC$, $R_{\text{eff}}$, and $ABHI$) with a bounded residual booster. This prevents runaway extrapolation errors while capturing complex non-linear electro-thermal transients.

---

## 4. Multi-Horizon Future SOH Forecasting

| Forecast Horizon | Steps Ahead | Holdout Test MAE (%) | Holdout Test RMSE (%) | Max Error (%) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Short-Term** | 30 steps (~1 cycle) | 4.570 | 5.907 | 15.288 | 5.114 |
| **Medium-Term** | 150 steps (~5 cycles) | 5.146 | 6.407 | 15.338 | 5.776 |
| **Long-Term** | 300 steps (~10 cycles) | 5.861 | 7.064 | 15.316 | 6.616 |

Forecast uncertainty naturally increases with prediction horizon, which is reflected in decreasing prediction confidence scores.

---

## 5. ABHI Weighting Formulations Comparison

| Formulation Method | Optimization Strategy | Validation MAE | Validation RMSE | Pearson Correlation |
| :--- | :--- | :--- | :--- | :--- |
| **Equal Weights** | Fixed uniform $w_i = 0.20$ | 33.519 | 34.021 | 0.6605 |
| **Learned Constrained** | SLSQP quadratic programming on train ($w_i \ge 0.05$) | **15.834** | **19.384** | **0.6902** |
| **Context-Aware** | Dynamic weighting conditioned on thermal stress | 15.803 | 19.378 | 0.6895 |

**Conclusion:** The data-driven learned constrained formulation reduces validation error by **52.8%** compared to naive equal weights.
