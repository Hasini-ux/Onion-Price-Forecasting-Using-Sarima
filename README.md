# Onion Price Crisis Forecasting using SARIMA

Predicting India's onion price spikes using 25 years of monthly market data (2001–2025).

## Overview

India's onion market experiences recurring price crises every October–November, costing an estimated ₹5,000 Crores annually in volatility-related losses. The core problem is that government interventions happen reactively — after prices have already spiked.

This project applies SARIMA time series modelling to forecast the crisis window up to 3 months in advance, shifting the approach from crisis response to proactive inventory management.

**Forecast:** ₹5,200/quintal in October 2026 (95% confidence interval upper bound: ₹7,000/quintal)

## Tools Used

- Python — pandas, numpy, matplotlib, seaborn, statsmodels, scipy
- Models — SARIMA(1,1,1)×(0,1,1,12), Holt-Winters Triple Exponential Smoothing
- Statistical Tests — ADF Test, Ljung-Box Test, ACF/PACF Analysis
- Data Source — Agmarknet (Government of India)

## Files

| File | Description |
|---|---|
| `onion_analysis.py` | Full Python code — preprocessing, modelling, visualisations |
| `onion_monthly.csv` | Monthly modal onion prices, 2001–2025 |
| `FINAL_WORKING_PRESENTATION.html` | Interactive 26-slide presentation (open in browser) |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
python onion_analysis.py
```

This generates all visualisations automatically.

## Key Findings

- Price spikes follow a predictable 12-month biological cycle tied to the Kharif–Rabi harvest gap — not random shocks
- Baseline prices have been rising ~8% per year independent of seasonality (structural inflation)
- SARIMA outperformed Holt-Winters on MAPE (28.4% vs 29.3%) and provides confidence intervals for risk planning
- Proposed algorithmic buffer stock release protocol estimated to save NAFED ~₹1,200 Crores/year

## Methodology

Raw data → STL Decomposition → ADF Stationarity Test → Differencing → ACF/PACF Analysis → SARIMA Grid Search → Residual Diagnostics → Forecast with Confidence Intervals

## Authors

Hasini Gogula and Palem Balaji Koushik
Christ University, School of Business and Management
Course: Time Series Analysis and Forecasting (CIA-III)
Faculty Guide: Dr. Jerrin Issac Sam
