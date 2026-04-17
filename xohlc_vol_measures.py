"""
xohlc_vol_measures.py

Compute OHLC-based volatility measures for one or all symbols in a CSV file
and print summary statistics.  Optionally write the daily measures to a CSV.

This is a lightweight companion to xohlc_vol.py that skips the lead-lag
correlation and regression analysis.

Usage
-----
    python xohlc_vol_measures.py
    python xohlc_vol_measures.py --symbol SPY
    python xohlc_vol_measures.py --output vol_measures.csv
    python xohlc_vol_measures.py --file prices_ohlc.csv --output vol_measures.csv
"""

from __future__ import annotations
import argparse
import time
import pandas as pd

from ohlc_io import available_symbols, read_ohlc_csv, write_vol_measures
from ohlc_vol import OHLC_VOL_COLS, clean_label, compute_vol_measures
from vol_analysis import acf_table, summary_stats

# ── constants ─────────────────────────────────────────────────────────────────
DATA_FILE = "prices_ohlc.csv"
# Print autocorrelations of vol measures (lags 1–NLAGS).
SHOW_ACF = True
NLAGS    = 10
# ──────────────────────────────────────────────────────────────────────────────


def print_symbol_measures(daily: pd.DataFrame, symbol: str) -> None:
    ff = lambda x: f"{x:.3f}"
    print(f"\n{'='*70}")
    print(f"symbol: {symbol}  |  "
          f"{daily.index[0].date()} to {daily.index[-1].date()}  "
          f"({len(daily)} trading days)")
    print(f"{'='*70}")

    vol_cols = [c for c in OHLC_VOL_COLS if c in daily.columns
                and not c.startswith("neg_")]

    print("\nSummary statistics of annualised volatility measures\n")
    print(summary_stats(daily, vol_cols).to_string(float_format=ff))

    if SHOW_ACF:
        print(f"\nAutocorrelations of annualised volatility measures (lags 1–{NLAGS})\n")
        print(acf_table(daily, vol_cols, nlags=NLAGS).to_string(float_format=ff))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and summarise OHLC-based volatility measures."
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
        "--output", default=None, metavar="FILE",
        help="write computed vol measures to this CSV file",
    )
    args = parser.parse_args()

    symbols = (
        [args.symbol] if args.symbol is not None
        else available_symbols(args.file)
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"data file: {args.file}")
    print(f"symbols: {', '.join(symbols)}")

    all_daily: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            all_daily[sym] = compute_vol_measures(read_ohlc_csv(args.file, sym))
        except ValueError as exc:
            print(f"warning: {exc}")

    for sym, daily in all_daily.items():
        print_symbol_measures(daily, sym)

    if args.output and all_daily:
        write_vol_measures(all_daily, OHLC_VOL_COLS, args.output)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntime elapsed (s): {time.time() - t0:.2f}")
