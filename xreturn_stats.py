"""
xreturn_stats.py

Compute return and volatility statistics for a multi-symbol OHLC CSV file.
Intended to be run on both the original and a resampled prices file to compare
the preservation (or destruction) of statistical structure.

For each symbol prints:
  1. Summary statistics of return series and vol measures
  2. Autocorrelations of returns
  3. Autocorrelations of squared returns
  4. Autocorrelations of absolute returns
  5. Autocorrelations of OHLC-based vol measures (volatility clustering)
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd

from ohlc_io import available_symbols, read_ohlc_csv
from ohlc_vol import compute_vol_measures

DATA_FILE = "prices_ohlc.csv"
NLAGS     = 10   # default ACF lags

# Return series to include in summary stats and ACF tables.
RETURN_COLS = ["ret_adj_cc", "ret_co", "ret_oc"]

# Vol measures to include in summary stats and ACF tables.
VOL_COLS = [
    "vol_adj_cc_ann",
    "vol_parkinson_ann",
    "vol_gk_ann",
    "vol_rs_ann",
    "vol_yz_ann",
]

RETURN_LABELS = {
    "ret_adj_cc": "adj-cc",
    "ret_co":     "close-to-open",
    "ret_oc":     "open-to-close",
}

VOL_LABELS = {
    "vol_adj_cc_ann":    "adj-cc",
    "vol_parkinson_ann": "parkinson",
    "vol_gk_ann":        "garman-klass",
    "vol_rs_ann":        "rogers-satchell",
    "vol_yz_ann":        "yang-zhang",
}


def _summary(df: pd.DataFrame, cols: list[str], labels: dict[str, str]) -> pd.DataFrame:
    rows = {}
    for col in cols:
        s = df[col].dropna()
        rows[labels.get(col, col)] = {
            "n":       int(s.count()),
            "mean":    s.mean(),
            "sd":      s.std(),
            "skew":    s.skew(),
            "ex_kurt": s.kurt(),
            "min":     s.min(),
            "max":     s.max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _acf_table(df: pd.DataFrame, cols: list[str], labels: dict[str, str], nlags: int) -> pd.DataFrame:
    rows = {}
    for col in cols:
        s = df[col].dropna()
        rows[labels.get(col, col)] = {
            str(lag): s.autocorr(lag=lag) for lag in range(1, nlags + 1)
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns.name = "lag"
    return out


def print_symbol_stats(daily: pd.DataFrame, symbol: str, nlags: int) -> None:
    ff3 = lambda x: f"{x:9.3f}"
    ff4 = lambda x: f"{x:9.4f}"

    # squared and absolute return series for all return cols
    for col in RETURN_COLS:
        daily[col + "_sq"]  = daily[col] ** 2
        daily[col + "_abs"] = daily[col].abs()

    print(f"\n{'='*70}")
    print(f"symbol: {symbol}  |  "
          f"{daily.index[0].date()} to {daily.index[-1].date()}  "
          f"({len(daily)} trading days)")
    print(f"{'='*70}")

    # ── summary stats ──────────────────────────────────────────────────────
    print("\nSummary statistics — returns (%)\n")
    print(_summary(daily, RETURN_COLS, RETURN_LABELS).to_string(float_format=ff3))

    print("\nSummary statistics — annualised vol measures (%)\n")
    print(_summary(daily, VOL_COLS, VOL_LABELS).to_string(float_format=ff3))

    # ── ACF of returns ─────────────────────────────────────────────────────
    print(f"\nAutocorrelations of returns\n")
    print(_acf_table(daily, RETURN_COLS, RETURN_LABELS, nlags).to_string(float_format=ff4))

    # ── ACF of squared returns ─────────────────────────────────────────────
    print(f"\nAutocorrelations of squared returns\n")
    sq_labels  = {c + "_sq":  RETURN_LABELS[c] for c in RETURN_COLS if c in RETURN_LABELS}
    print(_acf_table(daily,
                     [c + "_sq"  for c in RETURN_COLS],
                     sq_labels, nlags).to_string(float_format=ff4))

    # ── ACF of absolute returns ────────────────────────────────────────────
    print(f"\nAutocorrelations of absolute returns\n")
    abs_labels = {c + "_abs": RETURN_LABELS[c] for c in RETURN_COLS if c in RETURN_LABELS}
    print(_acf_table(daily,
                     [c + "_abs" for c in RETURN_COLS],
                     abs_labels, nlags).to_string(float_format=ff4))

    # ── ACF of vol measures ────────────────────────────────────────────────
    print(f"\nAutocorrelations of vol measures\n")
    print(_acf_table(daily, VOL_COLS, VOL_LABELS, nlags).to_string(float_format=ff4))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute return and volatility statistics for an OHLC CSV."
    )
    parser.add_argument(
        "--symbol", default=None,
        help="ticker to analyse; omit to analyse all symbols",
    )
    parser.add_argument(
        "--files", nargs="+", default=[DATA_FILE], metavar="FILE",
        help=f"one or more OHLC CSV files to process (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--nlags", type=int, default=NLAGS,
        help=f"number of ACF lags (default: {NLAGS})",
    )
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    for file in args.files:
        symbols = [args.symbol] if args.symbol else available_symbols(file)
        print(f"\n{'#'*70}")
        print(f"# file: {file}")
        print(f"# symbols: {', '.join(symbols)}")
        print(f"{'#'*70}")

        all_daily: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                all_daily[sym] = compute_vol_measures(read_ohlc_csv(file, sym))
            except ValueError as exc:
                print(f"warning: {exc}")

        # ── cross-asset adj-cc return correlation matrix ───────────────────
        if len(all_daily) > 1:
            ret_df = pd.DataFrame({sym: df["ret_adj_cc"] for sym, df in all_daily.items()})
            print("\nCorrelation matrix of adj-close-to-close returns\n")
            print(ret_df.corr().to_string(float_format=lambda x: f"{x:8.4f}"))

        for sym, daily in all_daily.items():
            print_symbol_stats(daily, sym, args.nlags)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntime elapsed (s): {time.time() - t0:.2f}")
