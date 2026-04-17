"""
xohlc_vol_var.py

Extends xohlc_vol.py: runs the same descriptive analysis of daily OHLC-based
volatility measures, then fits VAR models with and without non-negativity
constraints on the lag coefficients.  One or all symbols are analysed.

Toggle flags (command line):
  --no-ols-var        skip the full unrestricted OLS VAR
  --no-nnls-var       skip the full non-negative NNLS VAR
  --no-ols-var-restr  skip the restricted OLS VAR (one predictor at a time)
  --no-nnls-var-restr skip the restricted NNLS VAR (one predictor at a time)
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd

from vector_autoreg import (
    companion_matrix, spectral_radius,
    build_regressors, build_regressors_single, forward_endog,
    fit_ols, fit_nnls_var, coef_to_lag_list, n_negative_lag_coefs,
    print_coef_comparison, print_rmse_comparison,
    print_restricted_coefs, print_restricted_rmse_tables,
    print_ic_comparison,
)

from xohlc_vol import (
    available_symbols, read_ohlc_csv, compute_vol_measures,
    OHLC_VOL_COLS,
    clean_label, print_results,
    DATA_FILE,
)

# When True, use unadjusted close-to-close vol (vol_cc_ann) instead of
# dividend/split-adjusted (vol_adj_cc_ann) as the primary VAR variable.
INCLUDE_UNADJ_CC = False


# ---------------------------------------------------------------------------
# VAR fit and print  (mirrors xintraday_vol_var.fit_and_print_var)
# ---------------------------------------------------------------------------

def fit_and_print_var(
    daily: pd.DataFrame,
    var_cols: list[str],
    p: int,
    forward_horizons: list[int],
    fit_ols_var: bool,
    fit_nnls_var_: bool,
    fit_ols_var_restr: bool,
    fit_nnls_var_restr: bool,
) -> None:
    """Fit requested full and restricted VAR models for each forward horizon."""
    if not any([fit_ols_var, fit_nnls_var_, fit_ols_var_restr, fit_nnls_var_restr]):
        return

    var_labels = [clean_label(c) for c in var_cols]
    Y = daily[var_cols].dropna().values
    T, k = Y.shape

    _, exog_full = build_regressors(Y, p)
    exog_single = [
        build_regressors_single(Y, p, j)[1] for j in range(k)
    ]

    print(f"\nVAR({p}) fit to {k} volatility measures, T={T} obs")

    for h in forward_horizons:
        endog_h = forward_endog(Y, p, h)
        n_h = len(endog_h)
        exog_f_h = exog_full[:n_h]

        print(f"\n{'='*70}")
        print(f"Forward horizon: {h} day{'s' if h > 1 else ''}")
        print(f"{'='*70}")

        # ---- Full models ---------------------------------------------------
        full_models: dict[str, np.ndarray] = {}
        if fit_ols_var or fit_nnls_var_:
            if fit_ols_var:
                full_models["OLS"] = fit_ols(endog_h, exog_f_h)
            if fit_nnls_var_:
                full_models["NNLS"] = fit_nnls_var(endog_h, exog_f_h)

            print("\nFull VAR")
            for name, coef in full_models.items():
                lag_list = coef_to_lag_list(coef, k, p)
                sr = spectral_radius(companion_matrix(lag_list))
                n_neg = n_negative_lag_coefs(coef)
                print(f"  {name}: spectral radius={sr:.4f}  negative lag coefs={n_neg}")

            print("\nFull VAR coefficients\n")
            print_coef_comparison(full_models, var_labels, p)

            print("\nFull VAR in-sample RMSE by equation\n")
            print_rmse_comparison(endog_h, exog_f_h, full_models, var_labels)

        # ---- Restricted models (one predictor at a time) -------------------
        if fit_ols_var_restr or fit_nnls_var_restr:
            restr_rmse: dict[str, np.ndarray] = {}
            if fit_ols_var_restr:
                restr_rmse["OLS"] = np.zeros((k, k))
            if fit_nnls_var_restr:
                restr_rmse["NNLS"] = np.zeros((k, k))

            print("\nRestricted VAR coefficients (one predictor at a time)\n")
            for j, pred_label in enumerate(var_labels):
                exog_j_h = exog_single[j][:n_h]
                ols_coef_j  = fit_ols(endog_h, exog_j_h)      if fit_ols_var_restr  else None
                nnls_coef_j = fit_nnls_var(endog_h, exog_j_h) if fit_nnls_var_restr else None
                if fit_ols_var_restr:
                    resid = endog_h - exog_j_h @ ols_coef_j
                    restr_rmse["OLS"][j] = np.sqrt((resid ** 2).mean(axis=0))
                if fit_nnls_var_restr:
                    resid = endog_h - exog_j_h @ nnls_coef_j
                    restr_rmse["NNLS"][j] = np.sqrt((resid ** 2).mean(axis=0))
                print_restricted_coefs(ols_coef_j, nnls_coef_j, pred_label, var_labels, p)

            full_rmse: dict[str, np.ndarray] = {
                name: np.sqrt(((endog_h - exog_f_h @ coef) ** 2).mean(axis=0))
                for name, coef in full_models.items()
            }
            print_restricted_rmse_tables(restr_rmse, full_rmse, var_labels)

            print("\nInformation criteria: full VAR vs best single-predictor restricted VAR")
            print_ic_comparison(endog_h, exog_f_h, full_models, restr_rmse, var_labels, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute OHLC-based daily vol measures (same as xohlc_vol.py) then "
            "fit OLS and/or non-negative VAR models to them."
        )
    )
    parser.add_argument(
        "--symbol", default=None,
        help="ticker symbol to analyse; omit to analyse all symbols in the file",
    )
    parser.add_argument(
        "--file", default=DATA_FILE,
        help=f"path to OHLC CSV file (default: {DATA_FILE})",
    )
    parser.add_argument("--nlags", type=int, default=20,
                        help="lags for ACF and lead-lag correlation tables (default: 20)")
    parser.add_argument("--head", type=int, default=10,
                        help="rows to show in the daily measures table (default: 10)")
    parser.add_argument("--print-daily", action="store_true",
                        help="print the daily OHLC and volatility measures table")
    parser.add_argument("--nlags-var", type=int, default=None,
                        help="VAR lag order (overrides the nlags_var variable in the script)")
    parser.add_argument("--ols-var", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="fit full unrestricted OLS VAR (default: on)")
    parser.add_argument("--nnls-var", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="fit full non-negative NNLS VAR (default: on)")
    parser.add_argument("--ols-var-restr", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="fit restricted OLS VAR, one predictor at a time (default: on)")
    parser.add_argument("--nnls-var-restr", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="fit restricted NNLS VAR, one predictor at a time (default: on)")
    args = parser.parse_args()

    symbols = (
        [args.symbol] if args.symbol is not None
        else available_symbols(args.file)
    )

    nlags_var = 5   # default VAR lag order; overridden by --nlags-var
    if args.nlags_var is not None:
        nlags_var = args.nlags_var

    lag_target_cols = (
        ["vol_cc_ann", "vol_parkinson_ann"] if INCLUDE_UNADJ_CC
        else ["vol_adj_cc_ann", "vol_parkinson_ann"]
    )
    forward_horizons = [1, 5, 21]

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.6f}".format)

    print(f"data file: {args.file}")
    print(f"symbols: {', '.join(symbols)}")

    for symbol in symbols:
        df_raw = read_ohlc_csv(args.file, symbol)
        print(f"\n{'='*70}")
        print(f"symbol: {symbol}  |  {df_raw.index[0].date()} to {df_raw.index[-1].date()}  ({len(df_raw)} trading days)")
        print(f"{'='*70}")

        daily = compute_vol_measures(df_raw)

        print_results(
            daily,
            vol_cols=OHLC_VOL_COLS,
            nlags=args.nlags,
            head=args.head,
            print_daily_measures=args.print_daily,
            lag_target_cols=lag_target_cols,
            forward_horizons=forward_horizons,
        )

        fit_and_print_var(
            daily,
            var_cols=OHLC_VOL_COLS,
            p=nlags_var,
            forward_horizons=forward_horizons,
            fit_ols_var=args.ols_var,
            fit_nnls_var_=args.nnls_var,
            fit_ols_var_restr=args.ols_var_restr,
            fit_nnls_var_restr=args.nnls_var_restr,
        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntime elapsed (s): {time.time() - t0:.2f}")
