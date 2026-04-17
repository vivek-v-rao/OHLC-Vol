"""
xohlc_vol.py

Study OHLC-based volatility estimators and their ability to predict future
realised volatility.  Analyses one or all symbols in the input CSV.

Input file: prices_ohlc.csv  (multi-symbol, multi-field)
  Row 0: symbol label repeated for each field column
  Row 1: field names (Open, High, Low, Close, Adj Close)
  Row 2: "Date" sentinel row (skipped)
  Rows 3+: date-indexed data
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd

from ohlc_io import available_symbols, read_ohlc_csv, write_vol_measures
from ohlc_vol import OHLC_VOL_COLS, clean_label, compute_vol_measures
from vol_analysis import (
    acf_table, build_lag_predictors, build_lwma_predictors,
    build_ma_predictors, forward_vol_series, open_close_correlation_table,
    summary_stats, vol_correlation_matrix, vol_lead_lag_table,
    vol_lwma_table, vol_ma_table,
)
from nnls_reg import (fit_and_print_nonneg, fit_and_print_sqrt_var,
                      fit_and_print_log_vol, fit_and_print_var_space)

# ── constants ─────────────────────────────────────────────────────────────────
DATA_FILE = "prices_ohlc.csv"
# Set True to run NNLS regressions for each correlation table.
NONNEG_REG = True
# Individual toggles for which predictor types are used in regressions.
# All three default to True; ignored when NONNEG_REG is False.
NONNEG_REG_LAG  = False   # lagged measures (L1, L2, …)
NONNEG_REG_MA   = False   # simple moving averages
NONNEG_REG_LWMA = True   # linearly-weighted moving averages
# When True, at most one lag/window per vol measure is allowed as a predictor
# (the one with the highest absolute correlation with the target).
NONNEG_ONE_PER_GROUP = True
# When True, print std-err and t-stat alongside each coefficient.
NONNEG_SHOW_SE = True
# Minimum absolute t-stat for a predictor to remain in the regression.
# Set to None or 0 to disable.
NONNEG_T_STAT_MIN: float | None = None
# Maximum number of non-constant predictors.  Set to None to disable.
NONNEG_MAX_PREDS: int | None = None
# Vol measures to use as dependent variables.  None → use all OHLC_VOL_COLS.
DEP_VOL_COLS: list[str] | None = ["vol_adj_cc_ann"]
# Extra predictor columns (not in OHLC_VOL_COLS) for lead-lag tables and
# regressions; excluded from summary stats / ACF / correlation tables.
EXTRA_PRED_COLS: list[str] = ["neg_ret_adj_cc"]
# When True, also include neg_ret_adj_cc_pos = max(neg_ret_adj_cc, 0) as a
# predictor (captures the leverage effect only on down days).  Set False to
# exclude it without touching EXTRA_PRED_COLS.
INCLUDE_NEG_RET_POS = False
# Symbols whose vol measures are added as additional predictors for every
# analysed asset.  Must be present in the CSV file.  [] to disable.
EXTERNAL_PRED_SYMBOLS: list[str] = []
# Subset regression method run after each main regression:
#   "none"     — disabled
#   "backward" — sequentially remove the predictor with the lowest |t-stat|
#   "best"     — best subset by BIC at each size (falls back to "backward"
#                if the active set exceeds BEST_SUBSET_MAX_PREDS)
SUBSET_METHOD = "none"
BEST_SUBSET_MAX_PREDS = 12
# Individual toggles for the fixed summary tables printed per symbol.
SHOW_SUMMARY_STATS = False
SHOW_ACF = False
SHOW_OC_CORR = False
SHOW_VOL_CORR_MATRIX = False
# When False, suppress the correlation tables (lead-lag, simple MA, LWMA)
# and only print regression results.
SHOW_CORR_TABLES = False
# Decimal places for coef, std-err, and t-stat columns in regression tables.
NONNEG_COEF_DECIMALS = 3
# When True, resample rows of each symbol's vol measures with replacement before
# running regressions.  Destroys temporal structure; R² near zero confirms that
# the real results reflect genuine persistence rather than spurious structure.
RESAMPLE_VOL = False
# When True, also run the nonlinear sqrt-var regression y ≈ √(c0 + Σ cⱼ·xⱼ²)
# after each enabled level regression.  Coefficients are on the variance scale;
# R² and residuals are on the vol scale and directly comparable to the level model.
NONNEG_REG_SQRT_VAR = True
# When True, also run the log-vol regression log(y) = c0 + Σ cⱼ·log(xⱼ) with
# Duan smearing back-transform.  Predictors with non-positive values are dropped.
# R², BIC, and residual stats are on the vol scale for direct comparison.
NONNEG_REG_LOG_VOL = True
# When True, also run the var-space regression y² = c0 + Σ cⱼ·xⱼ² (linear NNLS
# on squared quantities) with ratio smearing back-transform to vol scale.
NONNEG_REG_VAR_SPACE = True
# When True, show the correlation of each predictor with the dependent variable.
NONNEG_SHOW_CORR = True
# When True, print the correlation matrix of the response and non-zero predictors
# after each main regression table.
NONNEG_SHOW_PRED_CORR_MATRIX = False
# When True, append skew and excess kurtosis of residuals to the regression header.
NONNEG_SHOW_RESID_STATS = True
# When True, each vol measure is predicted only by its own past values
# (e.g. adj-close-to-close predicted only by lagged adj-close-to-close).
# When False (default), all vol measures and EXTRA_PRED_COLS are pooled.
# Can be overridden with --same-measure on the command line.
SAME_MEASURE_ONLY = False
# ──────────────────────────────────────────────────────────────────────────────


def _combined_predictors(
    builder_fn,
    daily: pd.DataFrame,
    pred_cols: list[str],
    nlags: int,
    external_daily: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Build predictor DataFrame from own asset plus any external assets.

    External columns are renamed from e.g. ``L3.parkinson`` to
    ``L3.SPY.parkinson`` so that one_per_group treats each
    (symbol, measure) pair as a separate group.
    """
    own = builder_fn(daily, pred_cols, nlags)
    if not external_daily:
        return own
    parts = [own]
    for sym, df_ext in external_daily.items():
        ext = builder_fn(df_ext, pred_cols, nlags)
        ext.columns = [
            f"{c.split('.', 1)[0]}.{sym}.{c.split('.', 1)[1]}"
            for c in ext.columns
        ]
        parts.append(ext)
    return pd.concat(parts, axis=1)


def print_results(
    daily: pd.DataFrame,
    vol_cols: list[str],
    nlags: int,
    head: int,
    print_daily_measures: bool,
    lag_target_cols: list[str],
    forward_horizons: list[int],
    extra_pred_cols: list[str] | None = None,
    external_daily: dict[str, pd.DataFrame] | None = None,
    symbol: str = "",
    same_measure_only: bool = False,
) -> list[dict]:
    ff = lambda x: f"{x:.3f}"
    all_pred_cols = vol_cols + (extra_pred_cols or [])
    results: list[dict] = []

    if print_daily_measures:
        print(f"\nDaily OHLC and volatility measures (first {head} rows)\n")
        print_cols = ["Open", "High", "Low", "Close", "Adj_Close"] + vol_cols
        print(daily[print_cols].head(head).to_string(float_format=ff))

    if SHOW_SUMMARY_STATS:
        print("\nSummary statistics of annualised volatility measures\n")
        print(summary_stats(daily, vol_cols).to_string(float_format=ff))

    if SHOW_ACF:
        print("\nAutocorrelations of annualised volatility measures\n")
        print(acf_table(daily, vol_cols, nlags=nlags).to_string(float_format=ff))

    if SHOW_OC_CORR:
        print("\nLead-lag correlations of ret_oc with ret_co\n")
        print(open_close_correlation_table(daily).to_string(float_format=ff))

    if SHOW_VOL_CORR_MATRIX:
        print("\nCorrelation matrix of volatility measures\n")
        print(vol_correlation_matrix(daily, vol_cols).to_string(float_format=ff))

    for target in lag_target_cols:
        label = clean_label(target)
        pred_cols = [target] if same_measure_only else all_pred_cols
        for h in forward_horizons:
            hdr = f"{label} {h}-day-ahead"
            fwd = forward_vol_series(daily, target, h)

            if SHOW_CORR_TABLES:
                tbl = vol_lead_lag_table(daily, target, pred_cols, nlags=nlags, horizon=h)
                print(f"\n{hdr}: correlation with lagged volatility measures\n")
                print(tbl.to_string(float_format=ff))
            _reg_kwargs = dict(
                one_per_group=NONNEG_ONE_PER_GROUP,
                t_stat_min=NONNEG_T_STAT_MIN,
                max_preds=NONNEG_MAX_PREDS,
                show_se=NONNEG_SHOW_SE,
                subset_method=SUBSET_METHOD,
                best_subset_max_preds=BEST_SUBSET_MAX_PREDS,
                coef_decimals=NONNEG_COEF_DECIMALS,
                show_corr=NONNEG_SHOW_CORR,
                show_pred_corr_matrix=NONNEG_SHOW_PRED_CORR_MATRIX,
                show_resid_stats=NONNEG_SHOW_RESID_STATS,
            )
            def _collect(r: dict | None) -> None:
                if r is not None:
                    r["symbol"] = symbol
                    results.append(r)

            if NONNEG_REG and NONNEG_REG_LAG:
                _preds = _combined_predictors(build_lag_predictors, daily, pred_cols, nlags, external_daily)
                _collect(fit_and_print_nonneg(fwd, _preds, f"{hdr} ~ lagged measures", **_reg_kwargs))
                if NONNEG_REG_SQRT_VAR:
                    _collect(fit_and_print_sqrt_var(fwd, _preds, f"{hdr} ~ lagged measures", **_reg_kwargs))
                if NONNEG_REG_LOG_VOL:
                    _collect(fit_and_print_log_vol(fwd, _preds, f"{hdr} ~ lagged measures", **_reg_kwargs))
                if NONNEG_REG_VAR_SPACE:
                    _collect(fit_and_print_var_space(fwd, _preds, f"{hdr} ~ lagged measures", **_reg_kwargs))

            if SHOW_CORR_TABLES:
                tbl = vol_ma_table(daily, target, pred_cols, nlags=nlags, horizon=h)
                print(f"\n{hdr}: correlation with simple moving averages of vol measures\n")
                print(tbl.to_string(float_format=ff))
            if NONNEG_REG and NONNEG_REG_MA:
                _preds = _combined_predictors(build_ma_predictors, daily, pred_cols, nlags, external_daily)
                _collect(fit_and_print_nonneg(fwd, _preds, f"{hdr} ~ simple MA measures", **_reg_kwargs))
                if NONNEG_REG_SQRT_VAR:
                    _collect(fit_and_print_sqrt_var(fwd, _preds, f"{hdr} ~ simple MA measures", **_reg_kwargs))
                if NONNEG_REG_LOG_VOL:
                    _collect(fit_and_print_log_vol(fwd, _preds, f"{hdr} ~ simple MA measures", **_reg_kwargs))
                if NONNEG_REG_VAR_SPACE:
                    _collect(fit_and_print_var_space(fwd, _preds, f"{hdr} ~ simple MA measures", **_reg_kwargs))

            if SHOW_CORR_TABLES:
                tbl = vol_lwma_table(daily, target, pred_cols, nlags=nlags, horizon=h)
                print(f"\n{hdr}: correlation with linearly weighted moving averages of vol measures\n")
                print(tbl.to_string(float_format=ff))
            if NONNEG_REG and NONNEG_REG_LWMA:
                _preds = _combined_predictors(build_lwma_predictors, daily, pred_cols, nlags, external_daily)
                _collect(fit_and_print_nonneg(fwd, _preds, f"{hdr} ~ LWMA measures", **_reg_kwargs))
                if NONNEG_REG_SQRT_VAR:
                    _collect(fit_and_print_sqrt_var(fwd, _preds, f"{hdr} ~ LWMA measures", **_reg_kwargs))
                if NONNEG_REG_LOG_VOL:
                    _collect(fit_and_print_log_vol(fwd, _preds, f"{hdr} ~ LWMA measures", **_reg_kwargs))
                if NONNEG_REG_VAR_SPACE:
                    _collect(fit_and_print_var_space(fwd, _preds, f"{hdr} ~ LWMA measures", **_reg_kwargs))
    return results


def _print_regression_summary(results: list[dict]) -> None:
    """Print a compact summary table comparing all regression models run."""
    df = pd.DataFrame(results)
    if df.empty:
        return
    # Reorder and rename columns for display
    col_order = ["symbol", "model", "label", "n", "r2", "resid_sd",
                 "resid_skew", "resid_kurt", "bic", "n_preds"]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.rename(columns={
        "r2": "R²", "resid_sd": "sd", "resid_skew": "skew",
        "resid_kurt": "ex_kurt", "bic": "BIC", "n_preds": "#preds",
    })
    print(f"\n{'='*70}")
    print("Regression model comparison summary")
    print(f"{'='*70}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute OHLC-based volatility measures and study their predictive ability."
    )
    parser.add_argument(
        "--symbol", default=None,
        help="ticker symbol to analyse; omit to analyse all symbols in the file",
    )
    parser.add_argument(
        "--file", default=DATA_FILE,
        help=f"path to OHLC CSV file (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--nlags", type=int, default=20,
        help="number of lags for ACF and lead-lag tables (default: 20)",
    )
    parser.add_argument(
        "--head", type=int, default=10,
        help="rows to show in the daily measures table (default: 10)",
    )
    parser.add_argument(
        "--print-daily", action="store_true",
        help="print the daily OHLC and volatility measures table",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="write computed vol measures to this CSV file",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", metavar="H",
        help="forward horizons in trading days (default: 1 5 21)",
    )
    parser.add_argument(
        "--external-symbols", nargs="+", metavar="SYM",
        help="symbols to use as external volatility predictors (overrides EXTERNAL_PRED_SYMBOLS)",
    )
    parser.add_argument(
        "--same-measure", action="store_true",
        help="predict each vol measure only from its own past values",
    )
    args = parser.parse_args()

    symbols = (
        [args.symbol] if args.symbol is not None
        else available_symbols(args.file)
    )

    lag_target_cols = DEP_VOL_COLS if DEP_VOL_COLS is not None else OHLC_VOL_COLS
    forward_horizons = args.horizons if args.horizons is not None else [1, 5, 21]
    external_pred_symbols = args.external_symbols if args.external_symbols is not None else EXTERNAL_PRED_SYMBOLS
    same_measure_only = args.same_measure or SAME_MEASURE_ONLY

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"data file: {args.file}")
    if RESAMPLE_VOL:
        print("*** RESAMPLE_VOL=True: rows resampled with replacement — temporal structure destroyed ***")
    print(f"symbols: {', '.join(symbols)}")
    print(f"predictor mode: {'same measure only' if same_measure_only else 'all measures'}")
    if external_pred_symbols:
        print(f"external predictors: {', '.join(external_pred_symbols)}")
    if NONNEG_REG:
        constraints = []
        if NONNEG_T_STAT_MIN:
            constraints.append(f"t_stat_min={NONNEG_T_STAT_MIN}")
        if NONNEG_MAX_PREDS:
            constraints.append(f"max_preds={NONNEG_MAX_PREDS}")
        if constraints:
            print(f"regression constraints: {', '.join(constraints)}")

    # Pre-load all symbols needed (target + external) to avoid re-reading files.
    all_symbols_needed = list(dict.fromkeys(symbols + external_pred_symbols))
    all_daily: dict[str, pd.DataFrame] = {}
    for sym in all_symbols_needed:
        try:
            all_daily[sym] = compute_vol_measures(read_ohlc_csv(args.file, sym))
        except ValueError as exc:
            print(f"warning: {exc}")

    external_daily = {s: all_daily[s] for s in external_pred_symbols if s in all_daily}

    output_frames: dict[str, pd.DataFrame] = {}
    all_results: list[dict] = []
    for symbol in symbols:
        if symbol not in all_daily:
            continue
        daily = all_daily[symbol]
        if RESAMPLE_VOL:
            idx = np.random.choice(len(daily), size=len(daily), replace=True)
            daily = pd.DataFrame(daily.values[idx], index=daily.index, columns=daily.columns)
        output_frames[symbol] = daily
        # exclude the symbol being analysed from its own external predictors
        ext = {s: df for s, df in external_daily.items() if s != symbol}
        print(f"\n{'='*70}")
        print(f"symbol: {symbol}  |  {daily.index[0].date()} to {daily.index[-1].date()}  ({len(daily)} trading days)")
        print(f"{'='*70}")
        extra_preds = list(EXTRA_PRED_COLS)
        if INCLUDE_NEG_RET_POS and "neg_ret_adj_cc_pos" not in extra_preds:
            extra_preds.append("neg_ret_adj_cc_pos")
        sym_results = print_results(
            daily,
            same_measure_only=same_measure_only,
            vol_cols=OHLC_VOL_COLS,
            nlags=args.nlags,
            head=args.head,
            print_daily_measures=args.print_daily,
            lag_target_cols=lag_target_cols,
            forward_horizons=forward_horizons,
            extra_pred_cols=extra_preds,
            external_daily=ext or None,
            symbol=symbol,
        )
        all_results.extend(sym_results)

    if all_results:
        _print_regression_summary(all_results)

    if args.output and output_frames:
        write_vol_measures(output_frames, OHLC_VOL_COLS, args.output)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntime elapsed (s): {time.time() - t0:.2f}")
