# OHLC-Vol

A Python toolkit for computing OHLC-based volatility estimators, studying their
statistical properties, and evaluating their ability to forecast future realized
volatility using non-negative least squares (NNLS) regression.

---

## Overview

Daily OHLC (Open, High, Low, Close) prices contain more information about
intraday volatility than close-to-close returns alone.  This project implements
a suite of classical OHLC estimators, compares them via in-sample regressions
and walk-forward out-of-sample evaluation, and experiments with several
functional forms for the vol-on-vol forecasting relationship.

---

## Volatility estimators

| Column | Estimator |
|---|---|
| `vol_adj_cc_ann` | Adjusted close-to-close |
| `vol_cc_ann` | Unadjusted close-to-close |
| `vol_parkinson_ann` | [Parkinson (1980)](https://www-2.rotman.utoronto.ca/~kan/3032/pdf/FinancialAssetReturns/Parkinson_JB_1980.pdf) |
| `vol_gk_ann` | [Garman–Klass (1980)](https://www-2.rotman.utoronto.ca/~kan/3032/pdf/FinancialAssetReturns/Garman_Klass_JB_1980.pdf) |
| `vol_rs_ann` | [Rogers–Satchell (1991)](https://projecteuclid.org/journals/annals-of-applied-probability/volume-1/issue-4/Estimating-Variance-From-High-Low-and-Closing-Prices/10.1214/aoap/1177005835.full) |
| `vol_yz_ann` | [Yang–Zhang (2000)](https://www.jstor.org/stable/10.1086/209650?seq=1) |
| `vol_meilijson_ann` | Meilijson (2011) |
| `vol_ht_ann` | Hodges–Tompkins (bias-corrected CC) |
| `vol_tr_ann` | True Range |
| `vol_on_rs_ann` | Overnight + Rogers–Satchell |
| `neg_ret_adj_cc` | Negative adjusted return (leverage proxy) |
| `neg_ret_adj_cc_pos` | `max(neg_ret_adj_cc, 0)` (down-day leverage) |

All estimators are annualized (multiplied by √252) and expressed as percentage
volatility.

---

## Regression models

Four functional forms are supported for predicting h-day-ahead realized
volatility from linearly-weighted moving averages (LWMA) of past estimators:

| Model | Specification | Notes |
|---|---|---|
| `level` | vol = c₀ + Σ cⱼ·xⱼ | Linear NNLS; fastest |
| `sqrt-var` | vol = √(c₀ + Σ cⱼ·xⱼ²) | Non-linear NLS; slowest |
| `log-vol` | log(vol) = c₀ + Σ cⱼ·log(xⱼ) | Log-log NNLS + Duan smearing |
| `var-space` | vol² = c₀ + Σ cⱼ·xⱼ² | Linear NNLS on squared quantities + ratio smearing |

R², residual statistics, and BIC are reported on the vol scale for all four
models, enabling direct comparison.  Non-negative constraints ensure
forecasts are always non-negative.  Predictors are selected via a t-statistic
pruning loop and an optional one-per-group filter that retains only the best
LWMA window for each estimator.

---

## Files

### Library modules

| File | Purpose |
|---|---|
| `ohlc_vol.py` | Compute all OHLC vol estimators from raw OHLC data |
| `ohlc_io.py` | Read and write the multi-symbol OHLC CSV format |
| `vol_analysis.py` | Summary stats, ACF tables, correlation matrices, predictor builders (lag, MA, LWMA), forward vol target (RMS) |
| `nnls_reg.py` | NNLS regression engine: fitting, pruning, SE, BIC, and the four model types |

### Analysis scripts

| Script | Purpose |
|---|---|
| `xohlc_vol.py` | Main analysis: in-sample NNLS regressions for each estimator and forecast horizon; regression model comparison summary |
| `xohlc_vol_oos.py` | Walk-forward out-of-sample evaluation; OOS R², RMSE, MAE, bias per model and symbol |
| `xohlc_vol_measures.py` | Lightweight script: compute and display vol measures with summary stats and ACF; optionally write to CSV |
| `xohlc_vol_var.py` | Extends `xohlc_vol.py` with VAR models (OLS and NNLS, full and restricted) |
| `xreturn_stats.py` | Return and vol summary statistics and autocorrelations; useful for comparing original vs. resampled data |
| `xresample_ohlc.py` | Bootstrap resampler: draws daily OHLC rows with replacement to create a null dataset that destroys temporal autocorrelation while preserving marginal distributions |
| `vector_autoreg.py` | VAR fitting library used by `xohlc_vol_var.py`: OLS and NNLS VAR estimation, companion matrix, RMSE and IC tables |

### Data

| File | Description |
|---|---|
| `prices_ohlc.csv` | Multi-symbol daily OHLC prices (SPY, QQQ, GLD, USO, HYG; 2010–2026) |

---

## Input data format

The CSV uses a two-row header: the first row repeats the ticker symbol for each
field; the second row contains field names (`Open`, `High`, `Low`, `Close`,
`Adj Close`).  The first column is the date index.  This is the format produced
by `yfinance` when downloading multiple tickers.

```
         SPY                          QQQ          ...
         Open   High    Low  Close  Adj Close  Open ...
2010-01-04  ...
```

---

## Usage

### Compute vol measures and print summary statistics

```bash
python xohlc_vol_measures.py
python xohlc_vol_measures.py --symbol GLD --output gld_vol.csv
```

### In-sample regression analysis

```bash
# All symbols, default horizons (1, 5, 21 days)
python xohlc_vol.py

# Single symbol, 21-day horizon, SPY as external predictor
python xohlc_vol.py --symbol QQQ --horizons 21 --external-symbols SPY

# Predict each measure only from its own past values (baseline)
python xohlc_vol.py --same-measure --horizons 21

# Write vol measures to CSV
python xohlc_vol.py --output vol_measures.csv
```

### Walk-forward out-of-sample evaluation

```bash
# Default: level model, all symbols, horizons 1 5 21, annual refit
python xohlc_vol_oos.py

# Compare level and var-space models, monthly refit
python xohlc_vol_oos.py --models level var-space --step 21

# Same-measure baseline OOS
python xohlc_vol_oos.py --same-measure --horizons 21

# With external SPY predictors
python xohlc_vol_oos.py --external-symbols SPY --horizons 21
```

### Resampling null check

```bash
# Create resampled prices and compare return statistics
python xresample_ohlc.py --input prices_ohlc.csv --output prices_ohlc_resampled.csv
python xreturn_stats.py --file prices_ohlc_resampled.csv
```

---

## Key options

Most options are available as both a constant at the top of each script (for
persistent defaults) and as a command-line argument (for one-off overrides).

### `xohlc_vol.py`

| Option | Default | Description |
|---|---|---|
| `--symbol SYM` | all | Analyse one symbol |
| `--horizons H [H ...]` | 1 5 21 | Forecast horizons in trading days |
| `--external-symbols SYM [...]` | none | Add another symbol's vol measures as predictors |
| `--same-measure` | off | Restrict predictors to the same estimator as the target |
| `--output FILE` | none | Write vol measures to CSV |

### `xohlc_vol_oos.py`

| Option | Default | Description |
|---|---|---|
| `--symbol SYM` | all | Analyse one symbol |
| `--horizons H [H ...]` | 1 5 21 | Forecast horizons |
| `--models MODEL [...]` | level | Models: `level` `sqrt-var` `log-vol` `var-space` |
| `--min-train N` | 504 | Minimum training days (expanding window) |
| `--step N` | 252 | Refit interval in trading days |
| `--external-symbols SYM [...]` | none | External predictor symbols |
| `--same-measure` | off | Restrict predictors to the same estimator |

---

## Dependencies

```
numpy
pandas
scipy
```

---

## Selected findings (SPY, QQQ, GLD, USO, HYG; 2010–2026)

- **Level model wins in-sample and OOS** for equity-like assets (SPY, QQQ, HYG,
  USO).  The variance-based models (sqrt-var, var-space) do not consistently
  improve on the simpler linear specification.
- **GLD is an exception**: variance-based models show a meaningful in-sample
  improvement (R² +0.03), consistent with gold volatility having more symmetric
  dynamics than equities.
- **OOS R² is 3–10 points below in-sample R²**, a typical degradation from
  predictor selection.
- **5-day horizon** forecasts best on average (OOS R² ≈ 0.45); 1-day forecasts
  are hardest (R² ≈ 0.25).
- **External SPY predictors** help USO OOS but hurt GLD and HYG, suggesting
  the cross-asset vol relationship is non-stationary.
- **Same-measure baseline** (predicting vol from its own past only) is
  substantially weaker than pooling all OHLC estimators, confirming that the
  other estimators add genuine predictive value.
