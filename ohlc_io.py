"""
ohlc_io.py

Read multi-symbol OHLC price data from the project CSV format.

CSV layout
----------
  Row 0   : symbol label repeated for each field column
  Row 1   : field names (Open, High, Low, Close, Adj Close)
  Row 2   : "Date" sentinel row (skipped via dropna)
  Rows 3+ : date-indexed data
"""

from __future__ import annotations
import os
import pandas as pd


def available_symbols(filename: str) -> list[str]:
    """Return the list of symbols present in the CSV file."""
    header = pd.read_csv(filename, nrows=0, header=[0, 1], index_col=0)
    return list(header.columns.get_level_values(0).unique())


def read_ohlc_csv(filename: str, symbol: str) -> pd.DataFrame:
    """Read an OHLC CSV and return a DataFrame with columns
    Open, High, Low, Close, Adj_Close for the requested symbol."""
    raw = pd.read_csv(filename, header=[0, 1], index_col=0)
    raw = raw.dropna(how="all")
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"
    symbols_available = raw.columns.get_level_values(0).unique().tolist()
    if symbol not in symbols_available:
        raise ValueError(
            f"Symbol '{symbol}' not found. Available: {symbols_available}"
        )
    df = raw[symbol].copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df = df.astype(float)
    df = df.sort_index()
    return df


def write_vol_measures(
    frames: dict[str, "pd.DataFrame"],
    vol_cols: list[str],
    path: str,
) -> None:
    """Write computed vol measures for one or more symbols to a wide CSV.

    Output has one row per date.  Columns are prefixed with the symbol name
    when more than one symbol is present, e.g. ``SPY.vol_adj_cc_ann``.
    The file is created or overwritten.
    """
    parts = []
    multi = len(frames) > 1
    for sym, df in frames.items():
        cols = [c for c in vol_cols if c in df.columns]
        sub = df[cols].copy()
        if multi:
            sub.columns = [f"{sym}.{c}" for c in sub.columns]
        parts.append(sub)
    out = pd.concat(parts, axis=1)
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")
    print(f"vol measures written to {path}  ({len(out)} rows, {len(out.columns)} columns)")
