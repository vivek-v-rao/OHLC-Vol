"""
xresample_ohlc.py

Create a resampled version of a multi-symbol OHLC CSV file.

Daily return tuples are resampled jointly (same row index for every symbol)
with replacement, destroying temporal autocorrelation while preserving the
marginal distribution of daily returns and intraday OHLC structure.

Output file is in the same multi-symbol CSV format as the input and can be
passed directly to xohlc_vol.py.

Usage
-----
    python xresample_ohlc.py [--input FILE] [--output FILE] [--seed N]
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd

# When True, run xreturn_stats.py on both the input and output files after
# writing the resampled CSV.
RUN_RETURN_STATS = True

from ohlc_io import available_symbols, read_ohlc_csv

INPUT_FILE  = "prices_ohlc.csv"
OUTPUT_FILE = "prices_ohlc_resampled.csv"


def _resample_symbol(
    df: pd.DataFrame,
    row_idx: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct OHLC prices for one symbol from resampled return tuples.

    Parameters
    ----------
    df      : DataFrame with columns Open, High, Low, Close, Adj_Close,
              sorted by date, length n.
    row_idx : integer array of length n-1, indexing into the n-1 available
              daily return tuples (from rows 1..n-1 of df).

    Returns
    -------
    DataFrame with the same index and columns as df.  Row 0 is kept as-is
    (price anchor); rows 1..n-1 are reconstructed from the resampled returns.
    """
    o  = df["Open"].values
    h  = df["High"].values
    l  = df["Low"].values
    c  = df["Close"].values
    ac = df["Adj_Close"].values
    n  = len(df)

    # ── extract return tuples from rows 1 … n-1 ───────────────────────────
    ret_co  = np.log(o[1:]  / c[:-1])   # overnight (close → open)
    ret_oc  = np.log(c[1:]  / o[1:])    # intraday  (open  → close)
    h_rel   = np.log(h[1:]  / o[1:])    # high  relative to open
    l_rel   = np.log(l[1:]  / o[1:])    # low   relative to open
    adj_ret = np.log(ac[1:] / ac[:-1])  # adj-close log-return

    # ── resample ───────────────────────────────────────────────────────────
    r_co  = ret_co [row_idx]
    r_oc  = ret_oc [row_idx]
    r_h   = h_rel  [row_idx]
    r_l   = l_rel  [row_idx]
    r_adj = adj_ret[row_idx]

    # ── vectorised reconstruction ──────────────────────────────────────────
    # log(C_t) = log(C_0) + cumsum(r_co_t + r_oc_t)
    daily_cc = r_co + r_oc

    log_c       = np.empty(n)
    log_c[0]    = np.log(c[0])
    log_c[1:]   = log_c[0] + np.cumsum(daily_cc)

    log_o       = np.empty(n)
    log_o[0]    = np.log(o[0])
    log_o[1:]   = log_c[:-1] + r_co       # O_t = C_{t-1} * exp(r_co_t)

    log_h       = np.empty(n)
    log_h[0]    = np.log(h[0])
    log_h[1:]   = log_o[1:] + r_h

    log_l       = np.empty(n)
    log_l[0]    = np.log(l[0])
    log_l[1:]   = log_o[1:] + r_l

    log_ac      = np.empty(n)
    log_ac[0]   = np.log(ac[0])
    log_ac[1:]  = log_ac[0] + np.cumsum(r_adj)

    return pd.DataFrame({
        "Open":      np.exp(log_o),
        "High":      np.exp(log_h),
        "Low":       np.exp(log_l),
        "Close":     np.exp(log_c),
        "Adj Close": np.exp(log_ac),
    }, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample a multi-symbol OHLC CSV with replacement."
    )
    parser.add_argument(
        "--input", default=INPUT_FILE,
        help=f"input OHLC CSV file (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"output CSV file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    symbols = available_symbols(args.input)
    print(f"input:   {args.input}")
    print(f"symbols: {', '.join(symbols)}", flush=True)

    # ── load all symbols ───────────────────────────────────────────────────
    all_df: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        all_df[sym] = read_ohlc_csv(args.input, sym)

    # ── single joint resample index (same for all symbols) ─────────────────
    n = len(next(iter(all_df.values())))
    row_idx = np.random.choice(n - 1, size=n - 1, replace=True)

    # ── resample each symbol ───────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df_r = _resample_symbol(all_df[sym], row_idx)
        df_r.columns = pd.MultiIndex.from_product([[sym], df_r.columns])
        frames.append(df_r)

    combined = pd.concat(frames, axis=1)
    combined.index.name = "Date"

    # ── write output ───────────────────────────────────────────────────────
    combined.to_csv(args.output)
    print(f"output:  {args.output}  ({n} rows x {len(symbols)} symbols)",
          flush=True)

    if RUN_RETURN_STATS:
        cmd = [sys.executable, "xreturn_stats.py", "--files", args.input, args.output]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
