# AETHERIX BATTERY INTELLIGENCE ALGORITHM (ABIA)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

ABIA is an explainable, domain-informed battery intelligence framework designed for EV battery health monitoring, degradation forecasting, multi-factor risk assessment, and battery circularity decisions.

Instead of deploying generic black-box ML algorithms (such as unconstrained Random Forest or XGBoost), ABIA introduces a custom hybrid architecture combining:
1. **Electro-thermal physics proxies** ($R_{\text{eff}}$, $EFC$, Coulomb integration)
2. **Normalized Health Indices** (VSI, TSI, CSI, PSI, DGI)
3. **Optimized Aetherix Battery Health Index (ABHI)**
4. **Battery Health Momentum (BHM)**
5. **7-Dimensional Battery Behaviour Fingerprint**
6. **An Independent Deterministic Safety Engine**
7. **Explainable Hierarchical Circularity Decision Matrix**

---

## 1. Project Objective
Deliver an end-to-end intelligence system capable of computing:
1. **Current SOH** (State of Health)
2. **Future SOH** (Configurable short/medium/long-term horizons)
3. **Degradation Rate** (% SOH loss per 100 equivalent full cycles)
4. **Battery Health Score** (ABHI, 0–100)
5. **Battery Degradation Trend**
6. **Anomaly / Risk Level** (LOW, MEDIUM, HIGH, CRITICAL)
7. **Health Momentum** (Stable, Slowly Deteriorating, Rapidly Deteriorating, Temporarily Fluctuating)
8. **Battery Behaviour Fingerprint** (7-dimensional operational vector)
9. **Prediction Confidence** (Distributional & epistemic grounded percentage)
10. **Circularity Decision** (6 definitive pathways)

### Circularity Pathways:
- `CONTINUE USING`
- `NEEDS MONITORING`
- `REPAIR / SERVICE`
- `SECOND-LIFE SUITABLE`
- `RECYCLE`
- `HIGH-RISK / QUARANTINE`

---

## 2. Mathematical Formulations

### 2.1 Health Indices (0–100 Normalized Scale)
- **Voltage Stability Index (VSI):**
  $$VSI = 100 \times \exp\left( - \left[ \lambda_v \cdot \frac{\sigma(V)}{\mu(V)} + \gamma_v \cdot \left|\frac{dV}{dt}\right| \right] \right)$$
- **Thermal Stress Index (TSI):**
  $$TSI = 100 \times \left(1.0 - \text{clip}\left(\frac{\max(0, T - 35) + 0.5 \cdot \max(0, 15 - T) + \alpha_T |dT/dt|}{25.0}, 0.0, 1.0\right)\right)$$
- **Current Stress Index (CSI):**
  $$CSI = 100 \times \left(1.0 - \text{clip}\left( \frac{I_{\text{RMS}} + \beta_I \cdot \text{IQR}(I)}{I_{\text{stress\_ref}}}, 0.0, 1.0 \right)\right)$$
- **Power Stability Index (PSI):**
  $$PSI = 100 \times \left(1.0 - \text{clip}\left(\frac{\sigma(P)}{P_{\text{ref}}}, 0.0, 1.0\right)\right)$$
- **Degradation Index (DGI):**
  $$DGI = 100 \times \left[ \theta_c \left(\frac{Q_{\text{est}}}{Q_{\text{initial}}}\right) + (1 - \theta_c) \text{clip}\left( 1 - \frac{R_{\text{eff}} - R_0}{1.5 R_0}, 0, 1 \right) \right]$$

### 2.2 Aetherix Battery Health Index (ABHI)
$$ABHI = w_1 VSI + w_2 TSI + w_3 CSI + w_4 PSI + w_5 DGI, \quad \sum w_i = 1, \quad w_i \ge 0.05$$
Trained via SLSQP constrained quadratic optimization on training data only.

### 2.3 Circularity Suitability Score (CSS)
$$CSS = 0.35 \cdot \text{SOH} + 0.20 \cdot \text{Future\_SOH} + 0.15 \cdot \text{ABHI} + 0.15 \cdot (100 - \text{Risk}) + 0.15 \cdot \text{Confidence}$$

---

## 3. Independent Safety Engine & Overrides
Safety logic is decoupled from machine learning models. Any critical safety trigger overrides AI recommendations and forces:
$$\to \mathbf{HIGH-RISK\ /\ QUARANTINE}$$
even if the battery reports 99% SOH.

Active Rules:
- Over-temperature $\ge 60.0^\circ\text{C}$ or under-temperature $\le -20.0^\circ\text{C}$
- Rapid thermal runaway precursor: $|dT/dt| \ge 1.5^\circ\text{C}/\text{step}$
- Terminal voltage out-of-bounds ($< 2.20\text{V}$ or $> 4.35\text{V}$)
- Abnormal current spike ($> 150\text{A}$)
- Thermal stability collapse ($TSI < 15.0$)

---

## 4. Quickstart Guide

### 4.1 Environment Setup
```bash
# Using uv (fastest) or standard python
uv venv .venv
uv pip install -r requirements.txt
```

### 4.2 Automated Inspection of Any Battery Dataset
```bash
.venv/Scripts/python -m src.abia.data.run_inspection
```
Inspects column semantics, timestamps, sampling frequency, missing values, duplicates, and generates `reports/dataset_inspection_report.json` and `.md`.

### 4.3 Preprocessing & Zero-Leakage Splitting
```bash
.venv/Scripts/python -m src.abia.data.run_preprocessing
```
Processes raw data strictly preserving original files in `data/raw/` and saving clean chronological data in `data/processed/`.

### 4.4 Running the Complete Pipeline
```bash
.venv/Scripts/python -u -m src.abia.run_pipeline
```
Computes features, health indices, fits ABHI weights, evaluates baselines vs ABIA, forecasts future horizons, runs safety evaluations, and writes executive intelligence reports to `reports/`.

### 4.5 Launching the Interactive Web Dashboard
```bash
.venv/Scripts/python -m uvicorn src.abia.dashboard.app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000` in your web browser.

---

## 5. Verification & Test Suite
Run the comprehensive test suite with `pytest`:
```bash
.venv/Scripts/pytest -v
```
All 11 unit tests verify schema discovery, preprocessing immutability, temporal leakage prevention, bounded indices ($0-100$), safety overrides, and circularity decision logic.

---

## 6. Scientific Limitations & Disclaimers
1. **Research Prototype Status:** All thresholds (CSS limits, voltage limits, thermal limits) are configurable research prototype parameters. They do not constitute certified OEM automotive safety standards.
2. **Effective Resistance Proxy:** Termed an *effective resistance proxy* ($R_{\text{eff}} \approx \Delta V / \Delta I$), not true internal resistance, due to operating dynamic current excitation conditions.
3. **No 100% Accuracy Claims:** Realistic degradation models respect electrochemical variance and avoid artificially fabricated overfitted scores.
