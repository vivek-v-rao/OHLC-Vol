"""
vol_analysis.py

Analysis helpers for OHLC volatility studies: summary statistics,
autocorrelations, correlation matrices, lead-lag tables, and predictor
builders for use with regression models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ohlc_vol import clean_label


def summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sub = df[columns].rename(columns=clean_label)
    out = pd.DataFrame(index=sub.columns)
    out["n"]       = sub.count()
    out["mean"]    = sub.mean()
    out["sd"]      = sub.std()
    out["skew"]    = sub.skew()
    out["ex_kurt"] = sub.kurt()
    out["min"]     = sub.min()
    out["median"]  = sub.median()
    out["max"]     = sub.max()
    return out


def acf_table(df: pd.DataFrame, columns: list[str], nlags: int = 5) -> pd.DataFrame:
    rows = {}
    for col in columns:
        s = df[col].dropna()
        rows[clean_label(col)] = {str(lag): s.autocorr(lag=lag) for lag in range(1, nlags + 1)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns.name = "lag"
    return out


def open_close_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "ret_oc with previous-day ret_co": (df["ret_oc"], df["ret_co"].shift(1)),
        "ret_oc with next-day ret_co":     (df["ret_oc"], df["ret_co"].shift(-1)),
    }
    out = pd.DataFrame(index=pairs.keys(), columns=["corr"], dtype=float)
    for label, (x, y) in pairs.items():
        tmp = pd.concat([x, y], axis=1).dropna()
        out.loc[label, "corr"] = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
    return out


def vol_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    corr = df[columns].corr()
    labels = [clean_label(c) for c in columns]
    corr.index = labels
    corr.columns = labels
    return corr


def forward_vol_series(df: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """Root mean square of col over [t, t+horizon-1] — the h-day-ahead realized vol target.

    RMS = sqrt(mean(vol²)) is the standard realized-volatility definition:
    since vol_daily = sqrt(252) * |r|, this equals sqrt(252 * mean(r²)),
    consistent with how all OHLC estimators annualise daily variance.
    For h=1 RMS equals the daily value exactly.
    """
    return (df[col].pow(2)
                   .rolling(horizon).mean()
                   .shift(-(horizon - 1))
                   .pipe(np.sqrt))


def vol_lead_lag_table(
    df: pd.DataFrame, target_col: str, columns: list[str],
    nlags: int = 5, horizon: int = 1,
) -> pd.DataFrame:
    """Correlation of forward target with lag-k values of each estimator."""
    rows = {}
    target = forward_vol_series(df, target_col, horizon)
    for col in columns:
        corrs = {}
        for lag in range(1, nlags + 1):
            tmp = pd.concat([target, df[col].shift(lag)], axis=1).dropna()
            corrs[str(lag)] = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
        rows[clean_label(col)] = corrs
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns.name = "lag"
    return out


def vol_ma_table(
    df: pd.DataFrame, target_col: str, columns: list[str],
    nlags: int = 5, horizon: int = 1,
) -> pd.DataFrame:
    """Correlation of forward target with k-period simple MA of each estimator (ending at t-1)."""
    rows = {}
    target = forward_vol_series(df, target_col, horizon)
    for col in columns:
        corrs = {}
        for window in range(1, nlags + 1):
            ma = df[col].rolling(window).mean().shift(1)
            tmp = pd.concat([target, ma], axis=1).dropna()
            corrs[str(window)] = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
        rows[clean_label(col)] = corrs
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns.name = "window"
    return out


def vol_lwma_table(
    df: pd.DataFrame, target_col: str, columns: list[str],
    nlags: int = 5, horizon: int = 1,
) -> pd.DataFrame:
    """Correlation of forward target with k-period LWMA of each estimator (ending at t-1).

    Weights increase linearly from 1 (oldest) to k (most recent).
    """
    rows = {}
    target = forward_vol_series(df, target_col, horizon)
    for col in columns:
        corrs = {}
        for window in range(1, nlags + 1):
            weights = np.arange(1, window + 1, dtype=float)
            weights /= weights.sum()
            lwma = (
                df[col]
                .rolling(window)
                .apply(lambda x, w=weights: np.dot(x, w), raw=True)
                .shift(1)
            )
            tmp = pd.concat([target, lwma], axis=1).dropna()
            corrs[str(window)] = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
        rows[clean_label(col)] = corrs
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns.name = "window"
    return out


def build_lag_predictors(df: pd.DataFrame, columns: list[str], nlags: int) -> pd.DataFrame:
    """DataFrame of lagged predictors: L{lag}.{label} for each col and lag."""
    parts = {}
    for col in columns:
        for lag in range(1, nlags + 1):
            parts[f"L{lag}.{clean_label(col)}"] = df[col].shift(lag)
    return pd.DataFrame(parts, index=df.index)


def build_ma_predictors(df: pd.DataFrame, columns: list[str], nlags: int) -> pd.DataFrame:
    """DataFrame of simple-MA predictors: MA{window}.{label}"""
    parts = {}
    for col in columns:
        for window in range(1, nlags + 1):
            parts[f"MA{window}.{clean_label(col)}"] = df[col].rolling(window).mean().shift(1)
    return pd.DataFrame(parts, index=df.index)


def build_lwma_predictors(df: pd.DataFrame, columns: list[str], nlags: int) -> pd.DataFrame:
    """DataFrame of linearly-weighted-MA predictors: LWMA{window}.{label}"""
    parts = {}
    for col in columns:
        for window in range(1, nlags + 1):
            weights = np.arange(1, window + 1, dtype=float)
            weights /= weights.sum()
            lwma = (
                df[col]
                .rolling(window)
                .apply(lambda x, w=weights: np.dot(x, w), raw=True)
                .shift(1)
            )
            parts[f"LWMA{window}.{clean_label(col)}"] = lwma
    return pd.DataFrame(parts, index=df.index)
