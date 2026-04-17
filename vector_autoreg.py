"""
General-purpose vector autoregression (VAR) estimation and reporting.

Supports full VAR(p) and single-predictor restricted VAR models, with both
unrestricted OLS and non-negative-constrained (NNLS) fitting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def companion_matrix(coef_list: list[np.ndarray]) -> np.ndarray:
    """Build the VAR companion matrix from lag-1 … lag-p coefficient matrices."""
    k = coef_list[0].shape[0]
    p = len(coef_list)
    C = np.zeros((k * p, k * p))
    for i, A in enumerate(coef_list):
        C[:k, i * k:(i + 1) * k] = A
    if p > 1:
        C[k:, :-k] = np.eye(k * (p - 1))
    return C


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# ---------------------------------------------------------------------------
# Regressor construction
# ---------------------------------------------------------------------------

def build_regressors(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (endog, exog) for full VAR(p) with intercept.

    endog : (T-p, k)
    exog  : (T-p, 1 + k*p)  — first column is ones, then [Y_{t-1}, ..., Y_{t-p}]
    """
    T = len(Y)
    n = T - p
    lagged = np.hstack([Y[p - lag: T - lag] for lag in range(1, p + 1)])
    return Y[p:], np.hstack([np.ones((n, 1)), lagged])


def build_regressors_single(
    Y: np.ndarray, p: int, predictor_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (endog, exog) for VAR(p) restricted to one predictor variable.

    exog columns: [1, Y_{t-1,j}, Y_{t-2,j}, ..., Y_{t-p,j}]
    where j = predictor_idx.
    """
    T = len(Y)
    col = Y[:, predictor_idx]
    lagged = np.column_stack([col[p - lag: T - lag] for lag in range(1, p + 1)])
    return Y[p:], np.hstack([np.ones((T - p, 1)), lagged])


def forward_endog(Y: np.ndarray, p: int, h: int) -> np.ndarray:
    """Return h-day forward average of Y, aligned with lagged regressors.

    For horizon h, row i is mean(Y[p+i], ..., Y[p+i+h-1]), pairing with
    exog row i which contains lags of Y[p+i-1], ..., Y[p+i-p].
    Shape: (T - p - h + 1, k).  For h=1 this is just Y[p:].
    """
    base = Y[p:]
    n = len(base) - h + 1
    cs = np.concatenate([np.zeros((1, Y.shape[1])), base.cumsum(axis=0)], axis=0)
    return (cs[h:h + n] - cs[:n]) / h


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def fit_ols(endog: np.ndarray, exog: np.ndarray) -> np.ndarray:
    """Unrestricted OLS, equation by equation.  Returns (n_reg, k) coef matrix."""
    coef, _, _, _ = np.linalg.lstsq(exog, endog, rcond=None)
    return coef


def fit_nnls_var(endog: np.ndarray, exog: np.ndarray) -> np.ndarray:
    """Non-negative-constrained OLS, equation by equation.

    Returns (n_reg, k) coef matrix.  The intercept column is included in exog
    and is also constrained non-negative.
    """
    n_reg = exog.shape[1]
    k = endog.shape[1]
    coef = np.zeros((n_reg, k))
    for j in range(k):
        coef[:, j], _ = nnls(exog, endog[:, j])
    return coef


def coef_to_lag_list(coef: np.ndarray, k: int, p: int) -> list[np.ndarray]:
    """Extract p lag matrices (each k×k) from stacked (1+k*p, k) coef.

    Skips the first (intercept) row.
    """
    lag_part = coef[1:]
    return [lag_part[lag * k:(lag + 1) * k, :].T for lag in range(p)]


def n_negative_lag_coefs(coef: np.ndarray) -> int:
    """Count negative entries in the lag part of coef (excluding intercept row)."""
    return int((coef[1:] < 0).sum())


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def regressor_labels(var_labels: list[str], p: int) -> list[str]:
    """Return regressor labels: ['const', 'L1.v1', 'L1.v2', ..., 'Lp.vk']."""
    labels = ["const"]
    for lag in range(1, p + 1):
        for v in var_labels:
            labels.append(f"L{lag}.{v}")
    return labels


def coef_dataframe(coef: np.ndarray, var_labels: list[str], p: int) -> pd.DataFrame:
    """Wrap a (1+k*p, k) coef array in a labeled DataFrame."""
    return pd.DataFrame(
        coef,
        index=regressor_labels(var_labels, p),
        columns=var_labels,
    )


def rmse_series(
    endog: np.ndarray, exog: np.ndarray, coef: np.ndarray, var_labels: list[str]
) -> pd.Series:
    """Return per-equation RMSE as a Series indexed by var_labels."""
    resid = endog - exog @ coef
    return pd.Series(np.sqrt((resid ** 2).mean(axis=0)), index=var_labels)


def print_coef_comparison(
    models: dict[str, np.ndarray],
    var_labels: list[str],
    p: int,
) -> None:
    """Print full VAR coefficient tables one dependent variable at a time."""
    reg_labels = regressor_labels(var_labels, p)
    for j, dep_label in enumerate(var_labels):
        print(f"\n  Dependent variable: {dep_label}\n")
        data = {name: coef[:, j] for name, coef in models.items()}
        df = pd.DataFrame(data, index=reg_labels)
        print(df.to_string(float_format=lambda x: f"{x:9.4f}"))


def print_rmse_comparison(
    endog: np.ndarray,
    exog: np.ndarray,
    models: dict[str, np.ndarray],
    var_labels: list[str],
) -> None:
    """Print per-equation RMSE for each model; add NNLS-OLS diff if both present."""
    rmse_dict = {
        name: rmse_series(endog, exog, coef, var_labels)
        for name, coef in models.items()
    }
    df = pd.DataFrame(rmse_dict)
    if "OLS" in df.columns and "NNLS" in df.columns:
        df["NNLS-OLS"] = df["NNLS"] - df["OLS"]
    print(df.to_string(float_format=lambda x: f"{x:9.4f}"))


def print_restricted_coefs(
    ols_coef: np.ndarray | None,
    nnls_coef: np.ndarray | None,
    pred_label: str,
    var_labels: list[str],
    p: int,
) -> None:
    """Print restricted model coefficients with all dep vars as columns."""
    reg_labels = ["const"] + [f"L{lag}.{pred_label}" for lag in range(1, p + 1)]
    data = {}
    if ols_coef is not None:
        for j, v in enumerate(var_labels):
            data[(v, "OLS")] = ols_coef[:, j]
    if nnls_coef is not None:
        for j, v in enumerate(var_labels):
            data[(v, "NNLS")] = nnls_coef[:, j]
    df = pd.DataFrame(data, index=reg_labels)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    print(f"\n  Predictor: {pred_label}\n")
    print(df.to_string(float_format=lambda x: f"{x:8.4f}"))


def print_restricted_rmse_tables(
    restr_rmse: dict[str, np.ndarray],
    full_rmse: dict[str, np.ndarray],
    var_labels: list[str],
) -> None:
    """Print k×k RMSE matrix, best-predictor table, and full vs restricted summary."""
    for name, mat in restr_rmse.items():
        df = pd.DataFrame(mat, index=var_labels, columns=var_labels)
        print(f"\nRestricted {name} RMSE  (row=predictor, col=dependent variable)\n")
        print(df.to_string(float_format=lambda x: f"{x:8.4f}"))

    best_rows = {
        f"best {name} predictor": [var_labels[i] for i in mat.argmin(axis=0)]
        for name, mat in restr_rmse.items()
    }
    if best_rows:
        print("\nBest single predictor by dependent variable\n")
        print(pd.DataFrame(best_rows, index=var_labels).to_string())

    cmp = {}
    for name, rmse in full_rmse.items():
        cmp[f"full {name}"] = rmse
    for name, mat in restr_rmse.items():
        cmp[f"best restr {name}"] = mat.min(axis=0)
    if cmp:
        print("\nRMSE: full VAR vs best single-predictor restricted VAR\n")
        print(
            pd.DataFrame(cmp, index=var_labels)
            .to_string(float_format=lambda x: f"{x:8.4f}")
        )


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

def _ic_from_rmse(rmse: np.ndarray, n: int, m: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-equation AIC and BIC from RMSE values.

    AIC_i = n·ln(RMSE_i²) + 2·m_i  =  2·n·ln(RMSE_i) + 2·m_i
    BIC_i = n·ln(RMSE_i²) + ln(n)·m_i

    These are the standard Gaussian log-likelihood IC up to the constant
    n·(1 + ln(2π)) which cancels when comparing models with the same n.
    """
    log_term = 2.0 * n * np.log(np.asarray(rmse, dtype=float))
    aic = log_term + 2.0 * np.asarray(m, dtype=float)
    bic = log_term + np.log(n) * np.asarray(m, dtype=float)
    return aic, bic


def _nnls_nparams(coef: np.ndarray) -> np.ndarray:
    """Count non-zero parameters per equation (column) for a fitted NNLS model."""
    return (coef != 0).sum(axis=0).astype(float)


def print_ic_comparison(
    endog: np.ndarray,
    exog_full: np.ndarray,
    full_models: dict[str, np.ndarray],
    restr_rmse: dict[str, np.ndarray],
    var_labels: list[str],
    p: int,
) -> None:
    """Print per-equation AIC and BIC comparing full and best-restricted VAR models.

    Parameter counts:
      Full OLS  : 1 + k*p per equation (all regressors free)
      Full NNLS : effective non-zero parameters per equation
      Restr OLS : 1 + p per equation (intercept + p lags of one predictor)
      Restr NNLS: 1 + p per equation (upper bound; some lags may be zero)
    """
    if not full_models and not restr_rmse:
        return

    n, k = endog.shape
    m_full_ols = 1 + k * p       # params per equation, full OLS
    m_restr    = 1 + p           # params per equation, restricted (any method)

    aic_cols: dict[str, np.ndarray] = {}
    bic_cols: dict[str, np.ndarray] = {}

    # ---- Full models -------------------------------------------------------
    for name, coef in full_models.items():
        resid = endog - exog_full @ coef
        rmse_full = np.sqrt((resid ** 2).mean(axis=0))
        if name == "NNLS":
            m = _nnls_nparams(coef)
            label = "full NNLS (eff. params)"
        else:
            m = m_full_ols
            label = f"full {name}"
        aic_cols[label], bic_cols[label] = _ic_from_rmse(rmse_full, n, m)

    # ---- Best single-predictor restricted models ---------------------------
    for name, mat in restr_rmse.items():
        best_rmse = mat.min(axis=0)
        label = f"best restr {name}"
        aic_cols[label], bic_cols[label] = _ic_from_rmse(best_rmse, n, m_restr)

    def _winner(df: pd.DataFrame) -> pd.Series:
        return df.idxmin(axis=1).rename("best model (lowest IC)")

    print(f"\n  (n={n}, params: full OLS={m_full_ols}, restricted={m_restr})")
    print(f"  Full NNLS uses effective (non-zero) parameter count per equation.\n")

    aic_df = pd.DataFrame(aic_cols, index=var_labels)
    print("AIC by equation\n")
    print(pd.concat([aic_df, _winner(aic_df)], axis=1).to_string(
        float_format=lambda x: f"{x:10.1f}",
        formatters={"best model (lowest IC)": lambda x: x},
    ))

    print()
    bic_df = pd.DataFrame(bic_cols, index=var_labels)
    print("BIC by equation\n")
    print(pd.concat([bic_df, _winner(bic_df)], axis=1).to_string(
        float_format=lambda x: f"{x:10.1f}",
        formatters={"best model (lowest IC)": lambda x: x},
    ))
