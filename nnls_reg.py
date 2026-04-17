"""
nnls_reg.py

Non-negative least squares (NNLS) regression with iterative pruning.

Supports two pruning criteria (both optional, composable):
  t_stat_min  — drop the predictor with the lowest |t-stat| below the
                threshold and refit, until all non-zero coefficients pass.
  max_preds   — if more than max_preds non-constant predictors remain after
                t-stat pruning, drop the lowest-|t-stat| one and refit,
                repeating until the count is satisfied.

The intercept is never dropped by either criterion.
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import nnls as scipy_nnls
from scipy.optimize import least_squares as _scipy_ls

_SQRT_ZERO = 1e-10   # params below this are treated as zero in the sqrt-var model


def _print_nonneg_fit(
    label: str,
    n: int,
    r: float,
    r2: float,
    nonzero: list[tuple[str, float]],
    se_map: dict[str, float],
    bic_multi: float,
    best_bic1: float,
    best_name1: str,
    show_se: bool,
    coef_decimals: int = 6,
    corr_map: dict[str, float] | None = None,
    resid_stats: tuple[float, float, float, float, float, float] | None = None,
    response_stats: tuple[float, float, float, float, float, float] | None = None,
) -> None:
    """Print one fitted NNLS model (coefficients already sorted by caller)."""
    preferred = bic_multi < best_bic1
    w = coef_decimals + 6          # field width: decimals + sign/digits/dot
    fmt = f"{{:>{w}.{coef_decimals}f}}"
    if response_stats is not None:
        mean_y, sd_y, skew_y, kurt_y, min_y, max_y = response_stats
        print(f"\n  response: mean={mean_y:.3f}, sd={sd_y:.3f},"
              f" skew={skew_y:.3f}, ex_kurt={kurt_y:.3f},"
              f" min={min_y:.3f}, max={max_y:.3f}")
    print(f"\nNon-negative regression: {label}  (n={n}, R={r:.4f}, R²={r2:.4f})")
    if resid_stats is not None:
        mean_r, sd_r, skew_r, kurt_r, min_r, max_r = resid_stats
        print(f"  resid: mean={mean_r:.3f}, sd={sd_r:.3f},"
              f" skew={skew_r:.3f}, ex_kurt={kurt_r:.3f},"
              f" min={min_r:.3f}, max={max_r:.3f}")
    if nonzero:
        hdr = f"  {'predictor':<40s}  {'coef':>{w}s}"
        if show_se:
            hdr += f"  {'std-err':>{w}s}  {'t-stat':>{w}s}"
        if corr_map is not None:
            hdr += f"  {'corr':>{w}s}"
        print(hdr)
        for name, c in nonzero:
            row = f"  {name:<40s}  {fmt.format(c)}"
            if show_se:
                se = se_map.get(name, float("nan"))
                tstat = c / se if se and se != 0.0 else float("nan")
                row += f"  {fmt.format(se)}  {fmt.format(tstat)}"
            if corr_map is not None:
                corr_val = corr_map.get(name, float("nan"))
                row += f"  {fmt.format(corr_val)}"
            print(row)
    else:
        print("  (no non-zero coefficients)")
    pref_str = "multiple preferred" if preferred else "single preferred"
    print(f"  BIC: multiple={bic_multi:.1f}  best single={best_bic1:.1f}"
          f" ({best_name1})  [{pref_str}]")


def _bic_nnls(rss: float, n_nonzero: int, n: int) -> float:
    """BIC for an NNLS model: n·log(RSS/n) + k·log(n), k = # non-zero coefs."""
    if rss <= 0.0 or n <= 0:
        return -np.inf
    return n * np.log(rss / n) + n_nonzero * np.log(n)


def _nnls_se(
    X: np.ndarray, y: np.ndarray, coef: np.ndarray, ss_res: float, col_names: list[str]
) -> dict[str, float]:
    """OLS standard errors for the active (non-zero) set of an NNLS fit."""
    active_idx = np.where(coef > 0)[0]
    n = len(y)
    dof = n - len(active_idx)
    if dof <= 0 or len(active_idx) == 0:
        return {}
    X_act = X[:, active_idx]
    sigma2 = ss_res / dof
    se_vals = np.sqrt(sigma2 * np.diag(np.linalg.pinv(X_act.T @ X_act)))
    return {col_names[i]: float(s) for i, s in zip(active_idx, se_vals)}


def _series_stats(a: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return (mean, sd, skew, ex_kurt, min, max) of an array."""
    s = pd.Series(a)
    return float(s.mean()), float(s.std()), float(s.skew()), float(s.kurt()), float(s.min()), float(s.max())


def _resid_stats(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return (mean, sd, skew, ex_kurt, min, max) of residuals."""
    return _series_stats(y - y_hat)


def _abs_t(name: str, c: float, semap: dict[str, float]) -> float:
    se = semap.get(name, 0.0)
    return abs(c / se) if se > 0 else 0.0


def _fit_subset(
    y: np.ndarray,
    X_part: pd.DataFrame,
    subset: list[str],
    n: int,
    ss_tot: float,
    label: str,
    best_bic1: float,
    best_name1: str,
    show_se: bool,
    coef_decimals: int,
    corr_map: dict[str, float] | None = None,
    show_resid_stats: bool = False,
    response_stats: tuple[float, float, float, float, float, float] | None = None,
) -> float:
    """Fit NNLS on subset, print result, return BIC."""
    X_sub = np.column_stack([np.ones(n), X_part[subset].values])
    sub_names = ["const"] + subset
    sub_coef, _ = scipy_nnls(X_sub, y)
    sub_y_hat = X_sub @ sub_coef
    sub_ss_res = float(np.sum((y - sub_y_hat) ** 2))
    sub_r2 = 1.0 - sub_ss_res / ss_tot if ss_tot > 0 else 0.0
    sub_r = np.sqrt(max(0.0, sub_r2))
    sub_bic = _bic_nnls(sub_ss_res, int(np.sum(sub_coef > 0)), n)
    sub_se_map = _nnls_se(X_sub, y, sub_coef, sub_ss_res, sub_names)
    coef_map = dict(zip(sub_names, sub_coef))
    sub_nonzero = [(nm, coef_map[nm]) for nm in sub_names if coef_map[nm] > 0]
    sub_nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], sub_se_map))
    rs = _resid_stats(y, sub_y_hat) if show_resid_stats else None
    _print_nonneg_fit(label, n, sub_r, sub_r2, sub_nonzero, sub_se_map,
                      sub_bic, best_bic1, best_name1, show_se, coef_decimals,
                      corr_map=corr_map, resid_stats=rs, response_stats=response_stats)
    return sub_bic


def _run_backward_subset(
    y: np.ndarray,
    X_part: pd.DataFrame,
    n: int,
    ss_tot: float,
    start_preds: list[str],
    init_se_map: dict[str, float],
    init_coef_map: dict[str, float],
    best_bic1: float,
    best_name1: str,
    label: str,
    show_se: bool,
    coef_decimals: int,
    corr_map: dict[str, float] | None = None,
    show_resid_stats: bool = False,
    response_stats: tuple[float, float, float, float, float, float] | None = None,
) -> None:
    """Successively remove the predictor with the smallest |t-stat| and refit."""
    sub_active = list(start_preds)
    curr_se_map = init_se_map
    curr_coef_map = init_coef_map
    while len(sub_active) > 1:
        worst = min(
            sub_active,
            key=lambda nm: _abs_t(nm, curr_coef_map.get(nm, 0.0), curr_se_map),
        )
        sub_active.remove(worst)
        X_sub = np.column_stack([np.ones(n), X_part[sub_active].values])
        sub_names = ["const"] + sub_active
        sub_coef, _ = scipy_nnls(X_sub, y)
        sub_y_hat = X_sub @ sub_coef
        sub_ss_res = float(np.sum((y - sub_y_hat) ** 2))
        sub_r2 = 1.0 - sub_ss_res / ss_tot if ss_tot > 0 else 0.0
        sub_r = np.sqrt(max(0.0, sub_r2))
        sub_bic = _bic_nnls(sub_ss_res, int(np.sum(sub_coef > 0)), n)
        curr_se_map = _nnls_se(X_sub, y, sub_coef, sub_ss_res, sub_names)
        curr_coef_map = dict(zip(sub_names, sub_coef))
        sub_nonzero = [(nm, curr_coef_map[nm]) for nm in sub_names if curr_coef_map[nm] > 0]
        sub_nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], curr_se_map))
        rs = _resid_stats(y, sub_y_hat) if show_resid_stats else None
        _print_nonneg_fit(label, n, sub_r, sub_r2, sub_nonzero, curr_se_map,
                          sub_bic, best_bic1, best_name1, show_se, coef_decimals,
                          corr_map=corr_map, resid_stats=rs, response_stats=response_stats)


def _run_best_subset(
    y: np.ndarray,
    X_part: pd.DataFrame,
    n: int,
    ss_tot: float,
    start_preds: list[str],
    best_bic1: float,
    best_name1: str,
    label: str,
    show_se: bool,
    coef_decimals: int,
    max_size: int,
    corr_map: dict[str, float] | None = None,
    show_resid_stats: bool = False,
    response_stats: tuple[float, float, float, float, float, float] | None = None,
) -> None:
    """For each size from len(start_preds)-1 down to 1, find and print the
    best subset (lowest BIC) of that size.  Falls back to backward elimination
    if len(start_preds) > max_size."""
    if len(start_preds) > max_size:
        # too many candidates for exhaustive search — use backward instead
        X_cur = np.column_stack([np.ones(n), X_part[start_preds].values])
        cur_names = ["const"] + start_preds
        coef_arr, _ = scipy_nnls(X_cur, y)
        y_hat = X_cur @ coef_arr
        ss_res = float(np.sum((y - y_hat) ** 2))
        se_map = _nnls_se(X_cur, y, coef_arr, ss_res, cur_names)
        coef_map = dict(zip(cur_names, coef_arr))
        _run_backward_subset(y, X_part, n, ss_tot, start_preds,
                             se_map, coef_map,
                             best_bic1, best_name1, label, show_se, coef_decimals,
                             corr_map=corr_map, show_resid_stats=show_resid_stats,
                             response_stats=response_stats)
        return

    for size in range(len(start_preds) - 1, 0, -1):
        best_bic_s = np.inf
        best_subset_s: list[str] = []
        for subset in combinations(start_preds, size):
            X_sub = np.column_stack([np.ones(n), X_part[list(subset)].values])
            sub_coef, _ = scipy_nnls(X_sub, y)
            rss = float(np.sum((y - X_sub @ sub_coef) ** 2))
            bic = _bic_nnls(rss, int(np.sum(sub_coef > 0)), n)
            if bic < best_bic_s:
                best_bic_s = bic
                best_subset_s = list(subset)
        _fit_subset(y, X_part, best_subset_s, n, ss_tot, label,
                    best_bic1, best_name1, show_se, coef_decimals,
                    corr_map=corr_map, show_resid_stats=show_resid_stats,
                    response_stats=response_stats)


def fit_and_print_nonneg(
    y_series: pd.Series,
    X_df: pd.DataFrame,
    label: str,
    one_per_group: bool = False,
    t_stat_min: float | None = None,
    max_preds: int | None = None,
    show_se: bool = False,
    subset_method: str = "none",
    best_subset_max_preds: int = 12,
    coef_decimals: int = 6,
    show_corr: bool = False,
    show_pred_corr_matrix: bool = False,
    show_resid_stats: bool = False,
) -> dict | None:
    """NNLS regression of y on [const | X_df] with all coefficients ≥ 0.

    Parameters
    ----------
    y_series             : dependent variable
    X_df                 : candidate predictors (constant is added internally)
    label                : description printed in the header line
    one_per_group        : when True, column names must have the form
                           '{lag_desc}.{measure}'; only the column with the
                           highest |corr| to y is kept per measure group
    t_stat_min           : drop predictors with |t-stat| below this threshold
                           and refit iteratively; ignored when None or 0
    max_preds            : cap on non-constant predictors; excess are dropped
                           by lowest |t-stat|; ignored when None or 0
    show_se              : when True, print std-err and t-stat columns
    subset_method        : "none" — no subset regressions
                           "backward" — sequential removal by lowest |t-stat|
                           "best" — best-subset by BIC at each size
    best_subset_max_preds: fall back to "backward" if active set exceeds this
    coef_decimals        : decimal places for coef, std-err and t-stat columns
    show_corr            : when True, print a corr column (predictor vs y);
                           computed once and reused across all subset regressions
    show_pred_corr_matrix: when True, print the correlation matrix of y and the
                           non-zero predictors after the main regression table
    show_resid_stats     : when True, append skew and excess kurtosis of residuals
                           to the regression header line

    Returns a summary dict with model statistics (model, label, n, r2, resid_sd,
    resid_skew, resid_kurt, bic, n_preds), or None if there is insufficient data.
    """
    combined = pd.concat([y_series.rename("__y__"), X_df], axis=1).dropna()
    if len(combined) < 2:
        return None
    y = combined["__y__"].values
    X_part = combined.drop(columns="__y__")

    if one_per_group:
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for col in X_part.columns:
            if "." in col:
                grp = col.split(".", 1)[1]
                groups.setdefault(grp, []).append(col)
            else:
                ungrouped.append(col)
        best_cols = list(ungrouped)
        for grp, gcols in groups.items():
            corrs = [abs(float(np.corrcoef(y, X_part[c].values)[0, 1])) for c in gcols]
            best_cols.append(gcols[int(np.argmax(corrs))])
        X_part = X_part[best_cols]

    n = len(y)
    all_pred_names = list(X_part.columns)

    # ── compute response stats and predictor-to-target correlations once ──────
    response_stats = _series_stats(y) if show_resid_stats else None
    corr_map: dict[str, float] | None = None
    if show_corr:
        corr_map = {
            col: float(np.corrcoef(y, X_part[col].values)[0, 1])
            for col in X_part.columns
        }

    # ── iterative pruning (t-stat threshold and/or max predictor count) ───────
    active_pred_names = list(all_pred_names)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))

    while True:
        if active_pred_names:
            X_cur = np.column_stack([np.ones(n), X_part[active_pred_names].values])
        else:
            X_cur = np.ones((n, 1))
        cur_names = ["const"] + active_pred_names

        coef, _ = scipy_nnls(X_cur, y)
        y_hat = X_cur @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))

        if not prune:
            break

        se_map_iter = _nnls_se(X_cur, y, coef, ss_res, cur_names)
        active_idx = np.where(coef > 0)[0]

        n_active_preds = sum(1 for i in active_idx if cur_names[i] != "const")

        worst_name: str | None = None
        worst_abs_t = float("inf")
        for i in active_idx:
            name = cur_names[i]
            if name == "const":
                continue
            se = se_map_iter.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(coef[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t = abs_t
                worst_name = name

        if worst_name is None:
            break

        t_stat_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        count_fail  = bool(max_preds and max_preds > 0 and n_active_preds > max_preds)
        if not (t_stat_fail or count_fail):
            break
        active_pred_names.remove(worst_name)

    # ── final statistics ──────────────────────────────────────────────────────
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r  = np.sqrt(max(0.0, r2))

    bic_multi = _bic_nnls(ss_res, int(np.sum(coef > 0)), n)

    best_bic1 = np.inf
    best_name1 = ""
    for cname in all_pred_names:
        X1 = np.column_stack([np.ones(n), X_part[cname].values])
        c1, _ = scipy_nnls(X1, y)
        rss1 = float(np.sum((y - X1 @ c1) ** 2))
        bic1 = _bic_nnls(rss1, int(np.sum(c1 > 0)), n)
        if bic1 < best_bic1:
            best_bic1 = bic1
            best_name1 = cname

    preferred = bic_multi < best_bic1

    nonzero = [(name, c) for name, c in zip(cur_names, coef) if c > 0]

    # always compute SE so we can sort by |t-stat|
    se_map = _nnls_se(X_cur, y, coef, ss_res, cur_names)
    nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], se_map))

    rs = _resid_stats(y, y_hat) if show_resid_stats else None
    _print_nonneg_fit(label, n, r, r2, nonzero, se_map,
                      bic_multi, best_bic1, best_name1, show_se, coef_decimals,
                      corr_map=corr_map, resid_stats=rs, response_stats=response_stats)

    # ── build return dict ─────────────────────────────────────────────────────
    _resid_full = _series_stats(y - y_hat)   # (mean, sd, skew, ex_kurt, min, max)
    _result = {
        "model": "level",
        "label": label,
        "n": n,
        "r2": r2,
        "resid_sd": _resid_full[1],
        "resid_skew": _resid_full[2],
        "resid_kurt": _resid_full[3],
        "bic": bic_multi,
        "n_preds": sum(1 for nm, _ in nonzero if nm != "const"),
    }

    # ── predictor correlation matrix ──────────────────────────────────────────
    if show_pred_corr_matrix:
        nz_pred_names = [nm for nm, _ in nonzero if nm != "const"]
        if nz_pred_names:
            cm_df = pd.DataFrame({"y": y})
            for nm in nz_pred_names:
                cm_df[nm] = X_part[nm].values
            fmt_cm = lambda x: f"{x:.{coef_decimals}f}"
            print(f"\n  Correlation matrix (response and non-zero predictors)")
            print(cm_df.corr().to_string(float_format=fmt_cm))

    # ── subset regressions ────────────────────────────────────────────────────
    start_preds = [nm for nm, _ in nonzero if nm != "const"]
    if subset_method == "backward" and len(start_preds) > 1:
        _run_backward_subset(
            y, X_part, n, ss_tot, start_preds, se_map,
            dict(zip(cur_names, coef)),
            best_bic1, best_name1, label, show_se, coef_decimals,
            corr_map=corr_map, show_resid_stats=show_resid_stats,
            response_stats=response_stats,
        )
    elif subset_method == "best" and len(start_preds) > 1:
        _run_best_subset(
            y, X_part, n, ss_tot, start_preds,
            best_bic1, best_name1, label, show_se, coef_decimals,
            best_subset_max_preds,
            corr_map=corr_map, show_resid_stats=show_resid_stats,
            response_stats=response_stats,
        )
    return _result


# ── sqrt-var nonlinear model ───────────────────────────────────────────────────

def _sqrt_resid(params: np.ndarray, X_sq: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals y − √max(0, X_sq @ params) for scipy.optimize.least_squares."""
    return y - np.sqrt(np.maximum(X_sq @ params, 0.0))


def _sqrt_var_se(
    jac: np.ndarray, ss_res: float, n: int, n_active: int
) -> np.ndarray:
    """Parameter SEs from the Jacobian returned by least_squares.

    Uses the standard linearised-NLS approximation:
        Cov(params) ≈ σ² (JᵀJ)⁻¹,   σ² = ss_res / (n − n_active)
    """
    dof = n - n_active
    if dof <= 0 or jac.shape[1] == 0:
        return np.full(jac.shape[1], np.nan)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * (ss_res / dof)
        return np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        return np.full(jac.shape[1], np.nan)


def fit_and_print_sqrt_var(
    y_series: pd.Series,
    X_df: pd.DataFrame,
    label: str,
    one_per_group: bool = False,
    t_stat_min: float | None = None,
    max_preds: int | None = None,
    show_se: bool = False,
    subset_method: str = "none",
    best_subset_max_preds: int = 12,
    coef_decimals: int = 6,
    show_corr: bool = False,
    show_pred_corr_matrix: bool = False,
    show_resid_stats: bool = False,
) -> dict | None:
    """Non-negative nonlinear regression: y ≈ √(c0 + c1·x1² + c2·x2² + …).

    All coefficients are constrained ≥ 0 (so the radicand stays non-negative).
    Coefficients are on the variance scale (%²); the response and residuals are
    on the volatility scale (%), so R² and residual stats are directly comparable
    to the level NNLS model run on the same data.

    Warm-starts from the squared level-NNLS coefficients: if the level model
    gives ĉ_j, the initial variance-scale guess is ĉ_j².

    subset_method is accepted for signature compatibility but is not used.

    Returns a summary dict with model statistics, or None if there is insufficient data.
    """
    combined = pd.concat([y_series.rename("__y__"), X_df], axis=1).dropna()
    if len(combined) < 2:
        return None
    y = combined["__y__"].values
    X_part = combined.drop(columns="__y__")

    # one_per_group: filter on original (un-squared) correlations — same as level model
    if one_per_group:
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for col in X_part.columns:
            if "." in col:
                grp = col.split(".", 1)[1]
                groups.setdefault(grp, []).append(col)
            else:
                ungrouped.append(col)
        best_cols = list(ungrouped)
        for grp, gcols in groups.items():
            corrs = [abs(float(np.corrcoef(y, X_part[c].values)[0, 1])) for c in gcols]
            best_cols.append(gcols[int(np.argmax(corrs))])
        X_part = X_part[best_cols]

    n = len(y)
    all_pred_names = list(X_part.columns)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    response_stats = _series_stats(y) if show_resid_stats else None
    corr_map: dict[str, float] | None = None
    if show_corr:
        corr_map = {
            col: float(np.corrcoef(y, X_part[col].values)[0, 1])
            for col in X_part.columns
        }

    # ── warm start: level NNLS coefficients squared ───────────────────────────
    X_level = np.column_stack([np.ones(n), X_part.values])
    coef_level, _ = scipy_nnls(X_level, y)
    params_cur = coef_level ** 2          # [c0², c1², …] — variance scale

    # ── pruning loop (mirrors fit_and_print_nonneg) ───────────────────────────
    active_pred_names = list(all_pred_names)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))
    res = None

    while True:
        if active_pred_names:
            X_sq = np.column_stack(
                [np.ones(n), X_part[active_pred_names].values ** 2]
            )
        else:
            X_sq = np.ones((n, 1))
        cur_names = ["const"] + active_pred_names

        # trim / extend params_cur to match current column count
        k = X_sq.shape[1]
        if len(params_cur) >= k:
            p0 = np.maximum(params_cur[:k], 0.0)
        else:
            p0 = np.append(np.maximum(params_cur, 0.0), np.zeros(k - len(params_cur)))

        res = _scipy_ls(
            _sqrt_resid, p0, args=(X_sq, y), bounds=(0.0, np.inf), method="trf"
        )
        params_cur = res.x
        y_hat = np.sqrt(np.maximum(X_sq @ params_cur, 0.0))
        ss_res = float(np.sum((y - y_hat) ** 2))

        if not prune:
            break

        n_active = int(np.sum(params_cur > _SQRT_ZERO))
        se_arr   = _sqrt_var_se(res.jac, ss_res, n, n_active)
        se_map_iter = dict(zip(cur_names, se_arr))

        worst_name: str | None = None
        worst_abs_t = float("inf")
        n_active_preds = 0
        for i, name in enumerate(cur_names):
            if name == "const" or params_cur[i] <= _SQRT_ZERO:
                continue
            n_active_preds += 1
            se = se_map_iter.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(params_cur[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t = abs_t
                worst_name = name

        if worst_name is None:
            break

        t_stat_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        count_fail  = bool(max_preds and max_preds > 0 and n_active_preds > max_preds)
        if not (t_stat_fail or count_fail):
            break

        worst_idx = cur_names.index(worst_name)
        params_cur = np.delete(params_cur, worst_idx)
        active_pred_names.remove(worst_name)

    # ── final statistics ──────────────────────────────────────────────────────
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r  = np.sqrt(max(0.0, r2))
    bic_multi = _bic_nnls(ss_res, int(np.sum(params_cur > _SQRT_ZERO)), n)

    # best single-predictor BIC (sqrt-var model for each candidate)
    best_bic1  = np.inf
    best_name1 = ""
    for cname in all_pred_names:
        X1_sq  = np.column_stack([np.ones(n), X_part[cname].values ** 2])
        X1_lev = np.column_stack([np.ones(n), X_part[cname].values])
        c1, _  = scipy_nnls(X1_lev, y)
        r1     = _scipy_ls(_sqrt_resid, c1 ** 2, args=(X1_sq, y),
                           bounds=(0.0, np.inf), method="trf")
        y1     = np.sqrt(np.maximum(X1_sq @ r1.x, 0.0))
        rss1   = float(np.sum((y - y1) ** 2))
        bic1   = _bic_nnls(rss1, int(np.sum(r1.x > _SQRT_ZERO)), n)
        if bic1 < best_bic1:
            best_bic1  = bic1
            best_name1 = cname

    n_active = int(np.sum(params_cur > _SQRT_ZERO))
    se_arr   = _sqrt_var_se(res.jac, ss_res, n, n_active)
    se_map   = dict(zip(cur_names, se_arr))

    nonzero = [
        (name, float(params_cur[i]))
        for i, name in enumerate(cur_names)
        if params_cur[i] > _SQRT_ZERO
    ]
    nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], se_map))

    rs = _resid_stats(y, y_hat) if show_resid_stats else None
    _print_nonneg_fit(
        "sqrt-var: " + label, n, r, r2, nonzero, se_map,
        bic_multi, best_bic1, best_name1, show_se, coef_decimals,
        corr_map=corr_map, resid_stats=rs, response_stats=response_stats,
    )

    _resid_full = _series_stats(y - y_hat)
    _result = {
        "model": "sqrt-var",
        "label": label,
        "n": n,
        "r2": r2,
        "resid_sd": _resid_full[1],
        "resid_skew": _resid_full[2],
        "resid_kurt": _resid_full[3],
        "bic": bic_multi,
        "n_preds": sum(1 for nm, _ in nonzero if nm != "const"),
    }

    if show_pred_corr_matrix:
        nz_pred_names = [nm for nm, _ in nonzero if nm != "const"]
        if nz_pred_names:
            cm_df = pd.DataFrame({"y": y})
            for nm in nz_pred_names:
                cm_df[nm] = X_part[nm].values
            fmt_cm = lambda x: f"{x:.{coef_decimals}f}"
            print("\n  Correlation matrix (response and non-zero predictors)")
            print(cm_df.corr().to_string(float_format=fmt_cm))
    return _result


# ── log-vol model ─────────────────────────────────────────────────────────────

def fit_and_print_log_vol(
    y_series: pd.Series,
    X_df: pd.DataFrame,
    label: str,
    one_per_group: bool = False,
    t_stat_min: float | None = None,
    max_preds: int | None = None,
    show_se: bool = False,
    subset_method: str = "none",
    best_subset_max_preds: int = 12,
    coef_decimals: int = 6,
    show_corr: bool = False,
    show_pred_corr_matrix: bool = False,
    show_resid_stats: bool = False,
) -> dict | None:
    """NNLS in log-log space with Duan smearing back-transform to vol scale.

    Model: log(y) = c0 + c1·log(x1) + c2·log(x2) + …
    Coefficients are dimensionless log-space elasticities (≥ 0).
    Predictors with any non-positive values (e.g. neg-ret) are silently dropped.

    Predictions are back-transformed to the vol scale via Duan's smearing
    estimator: ŷ_vol = exp(ŷ_log) · mean(exp(ε̂_log)).  This corrects the
    downward bias of the naive exp(ŷ_log) without assuming normality of the
    log residuals.  R², BIC, and residual stats are computed on the vol scale
    and are directly comparable to the level and sqrt-var models.

    subset_method is accepted for signature compatibility but is not used.
    """
    combined = pd.concat([y_series.rename("__y__"), X_df], axis=1).dropna()
    if len(combined) < 2:
        return None
    y_vol = combined["__y__"].values
    X_raw = combined.drop(columns="__y__")

    # Drop predictors that cannot be log-transformed (zero or negative values)
    valid_cols = [c for c in X_raw.columns if (X_raw[c] > 0).all()]
    X_raw = X_raw[valid_cols]
    if len(valid_cols) == 0 or (y_vol <= 0).any():
        return None

    # Log-transform everything
    y     = np.log(y_vol)
    X_part = X_raw.apply(np.log)   # log-space predictor DataFrame

    n = len(y)
    ss_tot_vol = float(np.sum((y_vol - y_vol.mean()) ** 2))

    # one_per_group and corr_map use log-space quantities
    if one_per_group:
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for col in X_part.columns:
            if "." in col:
                grp = col.split(".", 1)[1]
                groups.setdefault(grp, []).append(col)
            else:
                ungrouped.append(col)
        best_cols = list(ungrouped)
        for grp, gcols in groups.items():
            corrs = [abs(float(np.corrcoef(y, X_part[c].values)[0, 1])) for c in gcols]
            best_cols.append(gcols[int(np.argmax(corrs))])
        X_part = X_part[best_cols]

    all_pred_names = list(X_part.columns)

    # response_stats shown on vol scale (same as other models)
    response_stats = _series_stats(y_vol) if show_resid_stats else None
    corr_map: dict[str, float] | None = None
    if show_corr:
        corr_map = {
            col: float(np.corrcoef(y, X_part[col].values)[0, 1])
            for col in X_part.columns
        }

    # ── pruning loop (identical structure to fit_and_print_nonneg) ─────────────
    active_pred_names = list(all_pred_names)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))

    while True:
        if active_pred_names:
            X_cur = np.column_stack([np.ones(n), X_part[active_pred_names].values])
        else:
            X_cur = np.ones((n, 1))
        cur_names = ["const"] + active_pred_names

        coef, _ = scipy_nnls(X_cur, y)
        y_hat_log = X_cur @ coef
        ss_res_log = float(np.sum((y - y_hat_log) ** 2))

        if not prune:
            break

        se_map_iter = _nnls_se(X_cur, y, coef, ss_res_log, cur_names)
        active_idx = np.where(coef > 0)[0]
        n_active_preds = sum(1 for i in active_idx if cur_names[i] != "const")

        worst_name: str | None = None
        worst_abs_t = float("inf")
        for i in active_idx:
            name = cur_names[i]
            if name == "const":
                continue
            se = se_map_iter.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(coef[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t = abs_t
                worst_name = name

        if worst_name is None:
            break

        t_stat_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        count_fail  = bool(max_preds and max_preds > 0 and n_active_preds > max_preds)
        if not (t_stat_fail or count_fail):
            break
        active_pred_names.remove(worst_name)

    # ── Duan smearing back-transform ──────────────────────────────────────────
    resid_log   = y - y_hat_log
    smearing    = float(np.mean(np.exp(resid_log)))
    y_hat_vol   = np.exp(y_hat_log) * smearing

    # ── vol-scale statistics ──────────────────────────────────────────────────
    resid_vol  = y_vol - y_hat_vol
    ss_res_vol = float(np.sum(resid_vol ** 2))
    r2 = 1.0 - ss_res_vol / ss_tot_vol if ss_tot_vol > 0 else 0.0
    r  = np.sqrt(max(0.0, r2))

    # BIC on vol scale so it is comparable to level and sqrt-var models
    bic_multi = _bic_nnls(ss_res_vol, int(np.sum(coef > 0)), n)

    # best single-predictor BIC (vol scale, with smearing)
    best_bic1  = np.inf
    best_name1 = ""
    for cname in all_pred_names:
        X1 = np.column_stack([np.ones(n), X_part[cname].values])
        c1, _ = scipy_nnls(X1, y)
        y1_log = X1 @ c1
        sm1    = float(np.mean(np.exp(y - y1_log)))
        y1_vol = np.exp(y1_log) * sm1
        rss1   = float(np.sum((y_vol - y1_vol) ** 2))
        bic1   = _bic_nnls(rss1, int(np.sum(c1 > 0)), n)
        if bic1 < best_bic1:
            best_bic1  = bic1
            best_name1 = cname

    se_map  = _nnls_se(X_cur, y, coef, ss_res_log, cur_names)
    nonzero = [(name, c) for name, c in zip(cur_names, coef) if c > 0]
    nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], se_map))

    rs = _resid_stats(y_vol, y_hat_vol) if show_resid_stats else None
    _print_nonneg_fit(
        "log-vol: " + label, n, r, r2, nonzero, se_map,
        bic_multi, best_bic1, best_name1, show_se, coef_decimals,
        corr_map=corr_map, resid_stats=rs, response_stats=response_stats,
    )

    _resid_full = _series_stats(y_vol - y_hat_vol)
    _result = {
        "model": "log-vol",
        "label": label,
        "n": n,
        "r2": r2,
        "resid_sd": _resid_full[1],
        "resid_skew": _resid_full[2],
        "resid_kurt": _resid_full[3],
        "bic": bic_multi,
        "n_preds": sum(1 for nm, _ in nonzero if nm != "const"),
    }

    if show_pred_corr_matrix:
        nz_pred_names = [nm for nm, _ in nonzero if nm != "const"]
        if nz_pred_names:
            cm_df = pd.DataFrame({"log_y": y})
            for nm in nz_pred_names:
                cm_df[nm] = X_part[nm].values
            fmt_cm = lambda x: f"{x:.{coef_decimals}f}"
            print("\n  Correlation matrix (log response and log non-zero predictors)")
            print(cm_df.corr().to_string(float_format=fmt_cm))
    return _result


# ── var-space model ───────────────────────────────────────────────────────────

_VAR_EPS = 1e-12   # floor for variance predictions before sqrt

def fit_and_print_var_space(
    y_series: pd.Series,
    X_df: pd.DataFrame,
    label: str,
    one_per_group: bool = False,
    t_stat_min: float | None = None,
    max_preds: int | None = None,
    show_se: bool = False,
    subset_method: str = "none",
    best_subset_max_preds: int = 12,
    coef_decimals: int = 6,
    show_corr: bool = False,
    show_pred_corr_matrix: bool = False,
    show_resid_stats: bool = False,
) -> dict | None:
    """NNLS in variance space with ratio smearing back-transform to vol scale.

    Model: y² = c0 + c1·x1² + c2·x2² + …  (linear NNLS on squared quantities)
    All predictors are squared before fitting; no predictors are dropped.

    Back-transform uses ratio smearing to correct the Jensen downward bias of
    √ŷ_var as an estimate of E[vol]:
        smearing = mean(vol_i / √max(ε, ŷ_var_i))
        vol_hat  = √ŷ_var · smearing

    R², BIC, and residual stats are computed on the vol scale and are directly
    comparable to the level, sqrt-var, and log-vol models.

    Compared to the sqrt-var nonlinear model this uses the same prediction
    formula but minimises variance-scale errors rather than vol-scale errors,
    and uses fast linear NNLS instead of an iterative nonlinear solver.

    subset_method is accepted for signature compatibility but is not used.
    """
    combined = pd.concat([y_series.rename("__y__"), X_df], axis=1).dropna()
    if len(combined) < 2:
        return None
    y_vol   = combined["__y__"].values
    X_raw   = combined.drop(columns="__y__")

    # Square response and all predictors
    y      = y_vol ** 2
    X_part = X_raw ** 2          # still a DataFrame; all values ≥ 0

    n = len(y)
    ss_tot_vol = float(np.sum((y_vol - y_vol.mean()) ** 2))

    # one_per_group and corr_map use var-space (squared) quantities
    if one_per_group:
        groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for col in X_part.columns:
            if "." in col:
                grp = col.split(".", 1)[1]
                groups.setdefault(grp, []).append(col)
            else:
                ungrouped.append(col)
        best_cols = list(ungrouped)
        for grp, gcols in groups.items():
            corrs = [abs(float(np.corrcoef(y, X_part[c].values)[0, 1])) for c in gcols]
            best_cols.append(gcols[int(np.argmax(corrs))])
        X_part = X_part[best_cols]

    all_pred_names = list(X_part.columns)

    response_stats = _series_stats(y_vol) if show_resid_stats else None
    corr_map: dict[str, float] | None = None
    if show_corr:
        corr_map = {
            col: float(np.corrcoef(y, X_part[col].values)[0, 1])
            for col in X_part.columns
        }

    # ── pruning loop ───────────────────────────────────────────────────────────
    active_pred_names = list(all_pred_names)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))

    while True:
        if active_pred_names:
            X_cur = np.column_stack([np.ones(n), X_part[active_pred_names].values])
        else:
            X_cur = np.ones((n, 1))
        cur_names = ["const"] + active_pred_names

        coef, _    = scipy_nnls(X_cur, y)
        y_hat_var  = X_cur @ coef
        ss_res_var = float(np.sum((y - y_hat_var) ** 2))

        if not prune:
            break

        se_map_iter    = _nnls_se(X_cur, y, coef, ss_res_var, cur_names)
        active_idx     = np.where(coef > 0)[0]
        n_active_preds = sum(1 for i in active_idx if cur_names[i] != "const")

        worst_name: str | None = None
        worst_abs_t = float("inf")
        for i in active_idx:
            name = cur_names[i]
            if name == "const":
                continue
            se = se_map_iter.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(coef[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t = abs_t
                worst_name  = name

        if worst_name is None:
            break

        t_stat_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        count_fail  = bool(max_preds and max_preds > 0 and n_active_preds > max_preds)
        if not (t_stat_fail or count_fail):
            break
        active_pred_names.remove(worst_name)

    # ── ratio smearing back-transform ─────────────────────────────────────────
    vol_naive  = np.sqrt(np.maximum(y_hat_var, _VAR_EPS))
    smearing   = float(np.mean(y_vol / vol_naive))
    y_hat_vol  = vol_naive * smearing

    # ── vol-scale statistics ──────────────────────────────────────────────────
    resid_vol  = y_vol - y_hat_vol
    ss_res_vol = float(np.sum(resid_vol ** 2))
    r2 = 1.0 - ss_res_vol / ss_tot_vol if ss_tot_vol > 0 else 0.0
    r  = np.sqrt(max(0.0, r2))

    bic_multi = _bic_nnls(ss_res_vol, int(np.sum(coef > 0)), n)

    # best single-predictor BIC (vol scale, with smearing)
    best_bic1  = np.inf
    best_name1 = ""
    for cname in all_pred_names:
        X1     = np.column_stack([np.ones(n), X_part[cname].values])
        c1, _  = scipy_nnls(X1, y)
        vn1    = np.sqrt(np.maximum(X1 @ c1, _VAR_EPS))
        sm1    = float(np.mean(y_vol / vn1))
        rss1   = float(np.sum((y_vol - vn1 * sm1) ** 2))
        bic1   = _bic_nnls(rss1, int(np.sum(c1 > 0)), n)
        if bic1 < best_bic1:
            best_bic1  = bic1
            best_name1 = cname

    se_map  = _nnls_se(X_cur, y, coef, ss_res_var, cur_names)
    nonzero = [(name, c) for name, c in zip(cur_names, coef) if c > 0]
    nonzero.sort(key=lambda t: -_abs_t(t[0], t[1], se_map))

    rs = _resid_stats(y_vol, y_hat_vol) if show_resid_stats else None
    _print_nonneg_fit(
        "var-space: " + label, n, r, r2, nonzero, se_map,
        bic_multi, best_bic1, best_name1, show_se, coef_decimals,
        corr_map=corr_map, resid_stats=rs, response_stats=response_stats,
    )

    _resid_full = _series_stats(y_vol - y_hat_vol)
    _result = {
        "model": "var-space",
        "label": label,
        "n": n,
        "r2": r2,
        "resid_sd": _resid_full[1],
        "resid_skew": _resid_full[2],
        "resid_kurt": _resid_full[3],
        "bic": bic_multi,
        "n_preds": sum(1 for nm, _ in nonzero if nm != "const"),
    }

    if show_pred_corr_matrix:
        nz_pred_names = [nm for nm, _ in nonzero if nm != "const"]
        if nz_pred_names:
            cm_df = pd.DataFrame({"y_sq": y})
            for nm in nz_pred_names:
                cm_df[nm] = X_part[nm].values
            fmt_cm = lambda x: f"{x:.{coef_decimals}f}"
            print("\n  Correlation matrix (var response and squared non-zero predictors)")
            print(cm_df.corr().to_string(float_format=fmt_cm))
    return _result
