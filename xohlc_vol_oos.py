"""
xohlc_vol_oos.py

Walk-forward out-of-sample evaluation of OHLC volatility forecasting models.

At each step the model is fitted on all data available up to that point
(expanding window), then applied to the next STEP_SIZE days.  OOS R²,
RMSE, MAE, and mean bias are reported per model type and symbol.

Model types
-----------
  level     — linear NNLS: vol ~ const + Σ cⱼ·xⱼ
  sqrt-var  — nonlinear:   vol ~ √(c0 + Σ cⱼ·xⱼ²)   [slow]
  log-vol   — log-log NNLS with Duan smearing back-transform
  var-space — linear NNLS on squared quantities with ratio smearing

Only level and var-space are enabled by default; enable the slower models
with --models sqrt-var log-vol (or set the toggles below).
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
from scipy.optimize import nnls as _scipy_nnls
from scipy.optimize import least_squares as _scipy_ls

from ohlc_io import available_symbols, read_ohlc_csv
from ohlc_vol import OHLC_VOL_COLS, clean_label, compute_vol_measures
from vol_analysis import build_lwma_predictors, forward_vol_series
from nnls_reg import _nnls_se, _SQRT_ZERO, _VAR_EPS, _sqrt_resid, _sqrt_var_se

# ── configuration ─────────────────────────────────────────────────────────────
DATA_FILE = "prices_ohlc.csv"
# Minimum number of training days before the first OOS prediction.
MIN_TRAIN_DAYS = 504        # 2 years
# Number of trading days between model refits (and the OOS window width).
STEP_SIZE = 252              # refit frequency
# LWMA window range — must match or exceed what xohlc_vol.py uses.
NLAGS = 20
# Dependent variable(s).  None → use all OHLC_VOL_COLS.
DEP_VOL_COLS: list[str] | None = ["vol_adj_cc_ann"]
# Extra predictor columns (same default as xohlc_vol.py).
EXTRA_PRED_COLS: list[str] = ["neg_ret_adj_cc"]
# External symbols whose vol measures are added as predictors.
EXTERNAL_PRED_SYMBOLS: list[str] = []

# Model type toggles (can be overridden with --models on the command line).
OOS_LEVEL     = True
OOS_SQRT_VAR  = False   # nonlinear solver — slow; ~10× slower per refit
OOS_LOG_VOL   = False   # also slower; enable explicitly if desired
OOS_VAR_SPACE = False

# Predictor selection options (mirror xohlc_vol.py defaults).
ONE_PER_GROUP = True        # keep best window per vol measure in each refit
T_STAT_MIN: float | None = None
MAX_PREDS: int | None = None
# When True, each vol measure is predicted only by its own past values
# (e.g. adj-close-to-close predicted only by lagged adj-close-to-close).
# When False (default), all vol measures and EXTRA_PRED_COLS are pooled.
# Can be overridden with --same-measure on the command line.
SAME_MEASURE_ONLY = False
# ──────────────────────────────────────────────────────────────────────────────

_ALL_MODELS = ["level", "sqrt-var", "log-vol", "var-space"]


# ── predictor helpers ─────────────────────────────────────────────────────────

def _combined_predictors(
    daily: pd.DataFrame,
    pred_cols: list[str],
    nlags: int,
    external_daily: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Build LWMA predictor DataFrame, appending external-symbol columns."""
    own = build_lwma_predictors(daily, pred_cols, nlags)
    if not external_daily:
        return own
    parts = [own]
    for sym, df_ext in external_daily.items():
        ext = build_lwma_predictors(df_ext, pred_cols, nlags)
        ext.columns = [
            f"{c.split('.', 1)[0]}.{sym}.{c.split('.', 1)[1]}"
            for c in ext.columns
        ]
        parts.append(ext)
    return pd.concat(parts, axis=1)


def _one_per_group(y: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Retain one column per vol-measure group (highest |corr| with y)."""
    groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for col in X.columns:
        if "." in col:
            grp = col.split(".", 1)[1]
            groups.setdefault(grp, []).append(col)
        else:
            ungrouped.append(col)
    best = list(ungrouped)
    for gcols in groups.values():
        corrs = [abs(float(np.corrcoef(y, X[c].values)[0, 1])) for c in gcols]
        best.append(gcols[int(np.argmax(corrs))])
    return X[best]


def _prune_nnls(
    y: np.ndarray,
    X_df: pd.DataFrame,
    t_stat_min: float | None,
    max_preds: int | None,
) -> tuple[np.ndarray, list[str]]:
    """NNLS with iterative t-stat / max-pred pruning.

    Returns (coef, cur_names) where cur_names = ["const"] + active_predictors.
    """
    n = len(y)
    active = list(X_df.columns)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))

    while True:
        X_cur = (np.column_stack([np.ones(n), X_df[active].values])
                 if active else np.ones((n, 1)))
        cur_names = ["const"] + active
        coef, _ = _scipy_nnls(X_cur, y)
        ss_res = float(np.sum((y - X_cur @ coef) ** 2))

        if not prune:
            break

        se_map = _nnls_se(X_cur, y, coef, ss_res, cur_names)
        active_idx = np.where(coef > 0)[0]
        n_active_preds = sum(1 for i in active_idx if cur_names[i] != "const")

        worst_name: str | None = None
        worst_abs_t = float("inf")
        for i in active_idx:
            name = cur_names[i]
            if name == "const":
                continue
            se = se_map.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(coef[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t, worst_name = abs_t, name

        if worst_name is None:
            break
        t_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        c_fail = bool(max_preds and max_preds > 0 and n_active_preds > max_preds)
        if not (t_fail or c_fail):
            break
        active.remove(worst_name)

    return coef, ["const"] + active


# ── model-specific fit-and-predict functions ──────────────────────────────────

def _fit_predict_level(
    y_tr: np.ndarray, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    one_per_group: bool, t_stat_min, max_preds,
) -> np.ndarray:
    if one_per_group:
        X_tr = _one_per_group(y_tr, X_tr)
    coef, names = _prune_nnls(y_tr, X_tr, t_stat_min, max_preds)
    pred_cols = [n for n in names if n != "const"]
    X_te_mat = np.column_stack([np.ones(len(X_te)), X_te[pred_cols].values])
    return np.maximum(X_te_mat @ coef, 0.0)


def _fit_predict_sqrt_var(
    y_tr: np.ndarray, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    one_per_group: bool, t_stat_min, max_preds,
) -> np.ndarray:
    if one_per_group:
        X_tr = _one_per_group(y_tr, X_tr)

    n_tr = len(y_tr)
    # warm start: square the level NNLS coefficients
    X_lev = np.column_stack([np.ones(n_tr), X_tr.values])
    coef_lev, _ = _scipy_nnls(X_lev, y_tr)
    params_cur = coef_lev ** 2

    active = list(X_tr.columns)
    prune = bool((t_stat_min and t_stat_min > 0) or (max_preds and max_preds > 0))

    while True:
        X_sq = (np.column_stack([np.ones(n_tr), X_tr[active].values ** 2])
                if active else np.ones((n_tr, 1)))
        cur_names = ["const"] + active
        k = X_sq.shape[1]
        if len(params_cur) >= k:
            p0 = np.maximum(params_cur[:k], 0.0)
        else:
            p0 = np.append(np.maximum(params_cur, 0.0), np.zeros(k - len(params_cur)))

        res = _scipy_ls(_sqrt_resid, p0, args=(X_sq, y_tr),
                        bounds=(0.0, np.inf), method="trf")
        params_cur = res.x
        y_hat = np.sqrt(np.maximum(X_sq @ params_cur, 0.0))
        ss_res = float(np.sum((y_tr - y_hat) ** 2))

        if not prune:
            break

        n_act = int(np.sum(params_cur > _SQRT_ZERO))
        se_arr = _sqrt_var_se(res.jac, ss_res, n_tr, n_act)
        se_map = dict(zip(cur_names, se_arr))

        worst_name: str | None = None
        worst_abs_t = float("inf")
        n_act_preds = 0
        for i, name in enumerate(cur_names):
            if name == "const" or params_cur[i] <= _SQRT_ZERO:
                continue
            n_act_preds += 1
            se = se_map.get(name, 0.0)
            if se <= 0:
                continue
            abs_t = abs(params_cur[i] / se)
            if abs_t < worst_abs_t:
                worst_abs_t, worst_name = abs_t, name

        if worst_name is None:
            break
        t_fail = bool(t_stat_min and t_stat_min > 0 and worst_abs_t < t_stat_min)
        c_fail = bool(max_preds and max_preds > 0 and n_act_preds > max_preds)
        if not (t_fail or c_fail):
            break
        wi = cur_names.index(worst_name)
        params_cur = np.delete(params_cur, wi)
        active.remove(worst_name)

    pred_cols = active
    X_te_sq = np.column_stack([np.ones(len(X_te)), X_te[pred_cols].values ** 2])
    return np.sqrt(np.maximum(X_te_sq @ params_cur, 0.0))


def _fit_predict_log_vol(
    y_tr: np.ndarray, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    one_per_group: bool, t_stat_min, max_preds,
) -> np.ndarray | None:
    # drop predictors that cannot be log-transformed in either split
    valid = [c for c in X_tr.columns if (X_tr[c] > 0).all() and (X_te[c] > 0).all()]
    if not valid or (y_tr <= 0).any():
        return None
    X_tr = X_tr[valid]
    X_te = X_te[valid]

    y_log = np.log(y_tr)
    X_log_tr = X_tr.apply(np.log)

    if one_per_group:
        X_log_tr = _one_per_group(y_log, X_log_tr)
        X_te = X_te[list(X_log_tr.columns)]

    coef, names = _prune_nnls(y_log, X_log_tr, t_stat_min, max_preds)
    pred_cols = [n for n in names if n != "const"]

    # Duan smearing factor from training residuals
    n_tr = len(y_tr)
    X_tr_mat = np.column_stack([np.ones(n_tr), X_log_tr[pred_cols].values])
    smearing = float(np.mean(np.exp(y_log - X_tr_mat @ coef)))

    X_te_mat = np.column_stack([np.ones(len(X_te)),
                                 np.log(X_te[pred_cols].values)])
    return np.exp(X_te_mat @ coef) * smearing


def _fit_predict_var_space(
    y_tr: np.ndarray, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    one_per_group: bool, t_stat_min, max_preds,
) -> np.ndarray:
    y_var = y_tr ** 2
    X_var_tr = X_tr ** 2

    if one_per_group:
        X_var_tr = _one_per_group(y_var, X_var_tr)

    coef, names = _prune_nnls(y_var, X_var_tr, t_stat_min, max_preds)
    pred_cols = [n for n in names if n != "const"]

    # ratio smearing from training fit
    n_tr = len(y_tr)
    X_tr_mat = np.column_stack([np.ones(n_tr), X_var_tr[pred_cols].values])
    vol_naive_tr = np.sqrt(np.maximum(X_tr_mat @ coef, _VAR_EPS))
    smearing = float(np.mean(y_tr / vol_naive_tr))

    X_te_mat = np.column_stack([np.ones(len(X_te)),
                                 X_te[pred_cols].values ** 2])
    vol_naive_te = np.sqrt(np.maximum(X_te_mat @ coef, _VAR_EPS))
    return vol_naive_te * smearing


# ── OOS statistics ─────────────────────────────────────────────────────────────

def _oos_stats(actuals: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    resid = actuals - preds
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "OOS-R²": r2,
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "MAE": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(preds - actuals)),
    }


# ── walk-forward loop ──────────────────────────────────────────────────────────

def walk_forward_oos(
    daily: pd.DataFrame,
    target_col: str,
    pred_cols: list[str],
    horizon: int,
    nlags: int,
    min_train: int,
    step: int,
    one_per_group: bool,
    t_stat_min: float | None,
    max_preds: int | None,
    model_types: list[str],
    external_daily: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward OOS evaluation.

    Returns a DataFrame with one row per model type and columns:
    model, n_oos, OOS-R², RMSE, MAE, bias.
    """
    X_all = _combined_predictors(daily, pred_cols, nlags, external_daily)
    fwd_all = forward_vol_series(daily, target_col, horizon)

    combined = pd.concat([fwd_all.rename("__y__"), X_all], axis=1).dropna()
    n_total = len(combined)

    if n_total <= min_train:
        return pd.DataFrame(columns=["model", "n_oos", "OOS-R²", "RMSE", "MAE", "bias"])

    preds_store: dict[str, list[float]] = {m: [] for m in model_types}
    actuals_store: list[float] = []

    _fit_predict = {
        "level":     _fit_predict_level,
        "sqrt-var":  _fit_predict_sqrt_var,
        "log-vol":   _fit_predict_log_vol,
        "var-space": _fit_predict_var_space,
    }

    t = min_train
    while t < n_total:
        test_end = min(t + step, n_total)
        y_tr = combined.iloc[:t]["__y__"].values
        X_tr = combined.iloc[:t].drop(columns="__y__")
        y_te = combined.iloc[t:test_end]["__y__"].values
        X_te = combined.iloc[t:test_end].drop(columns="__y__")

        actuals_store.extend(y_te.tolist())

        for m in model_types:
            try:
                p = _fit_predict[m](
                    y_tr, X_tr.copy(), X_te.copy(),
                    one_per_group, t_stat_min, max_preds,
                )
                if p is None:
                    preds_store[m].extend([float("nan")] * len(y_te))
                else:
                    preds_store[m].extend(p.tolist())
            except Exception:
                preds_store[m].extend([float("nan")] * len(y_te))

        t += step

    actuals = np.array(actuals_store)
    rows = []
    for m in model_types:
        preds = np.array(preds_store[m])
        mask = np.isfinite(preds) & np.isfinite(actuals)
        if mask.sum() < 2:
            rows.append({"model": m, "n_oos": 0,
                         "OOS-R²": float("nan"), "RMSE": float("nan"),
                         "MAE": float("nan"), "bias": float("nan")})
        else:
            stats = _oos_stats(actuals[mask], preds[mask])
            rows.append({"model": m, "n_oos": int(mask.sum()), **stats})

    return pd.DataFrame(rows)


# ── output ─────────────────────────────────────────────────────────────────────

def _print_oos_table(df: pd.DataFrame) -> None:
    ff = lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "   nan"
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward OOS evaluation of OHLC volatility forecasting models."
    )
    parser.add_argument("--symbol", default=None,
                        help="ticker to analyse; omit to analyse all symbols")
    parser.add_argument("--file", default=DATA_FILE,
                        help=f"OHLC CSV file (default: {DATA_FILE})")
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN_DAYS,
                        help=f"minimum training days (default: {MIN_TRAIN_DAYS})")
    parser.add_argument("--step", type=int, default=STEP_SIZE,
                        help=f"refit interval in trading days (default: {STEP_SIZE})")
    parser.add_argument("--horizons", type=int, nargs="+", metavar="H",
                        help="forecast horizons in trading days (default: 1 5 21)")
    parser.add_argument("--models", nargs="+", choices=_ALL_MODELS,
                        metavar="MODEL",
                        help="models to evaluate; choices: " + " ".join(_ALL_MODELS))
    parser.add_argument("--external-symbols", nargs="+", metavar="SYM",
                        help="external vol predictor symbols")
    parser.add_argument("--same-measure", action="store_true",
                        help="predict each vol measure only from its own past values")
    args = parser.parse_args()

    symbols = ([args.symbol] if args.symbol is not None
               else available_symbols(args.file))
    forward_horizons = args.horizons if args.horizons else [1, 5, 21]
    external_pred_syms = args.external_symbols or EXTERNAL_PRED_SYMBOLS

    if args.models:
        model_types = args.models
    else:
        model_types = (
            (["level"]     if OOS_LEVEL     else []) +
            (["sqrt-var"]  if OOS_SQRT_VAR  else []) +
            (["log-vol"]   if OOS_LOG_VOL   else []) +
            (["var-space"] if OOS_VAR_SPACE else [])
        )

    lag_target_cols = DEP_VOL_COLS if DEP_VOL_COLS is not None else OHLC_VOL_COLS
    all_pred_cols = OHLC_VOL_COLS + EXTRA_PRED_COLS
    same_measure_only = args.same_measure or SAME_MEASURE_ONLY

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print(f"data file: {args.file}")
    print(f"{len(symbols)} symbols: {' '.join(symbols)}")
    print(f"models: {', '.join(model_types)}")
    print(f"min_train={args.min_train} days, step={args.step} days")
    print(f"predictor mode: {'same measure only' if same_measure_only else 'all measures'}")
    if external_pred_syms:
        print(f"external predictors: {', '.join(external_pred_syms)}")

    # Pre-load all symbols
    all_symbols_needed = list(dict.fromkeys(symbols + external_pred_syms))
    all_daily: dict[str, pd.DataFrame] = {}
    for sym in all_symbols_needed:
        try:
            all_daily[sym] = compute_vol_measures(read_ohlc_csv(args.file, sym))
        except ValueError as exc:
            print(f"warning: {exc}")

    external_daily = {s: all_daily[s] for s in external_pred_syms if s in all_daily}

    all_rows: list[dict] = []   # for cross-symbol summary

    for symbol in symbols:
        if symbol not in all_daily:
            continue
        daily = all_daily[symbol]
        ext = {s: df for s, df in external_daily.items() if s != symbol}

        print(f"\n{'='*70}")
        print(f"symbol: {symbol}  |  "
              f"{daily.index[0].date()} to {daily.index[-1].date()}  "
              f"({len(daily)} trading days)")
        print(f"{'='*70}")

        for target in lag_target_cols:
            t_label = clean_label(target)
            eff_pred_cols = [target] if same_measure_only else all_pred_cols
            for h in forward_horizons:
                oos_df = walk_forward_oos(
                    daily=daily,
                    target_col=target,
                    pred_cols=eff_pred_cols,
                    horizon=h,
                    nlags=NLAGS,
                    min_train=args.min_train,
                    step=args.step,
                    one_per_group=ONE_PER_GROUP,
                    t_stat_min=T_STAT_MIN,
                    max_preds=MAX_PREDS,
                    model_types=model_types,
                    external_daily=ext or None,
                )
                n_oos = oos_df["n_oos"].max() if len(oos_df) else 0
                print(f"\n  {t_label} {h}-day-ahead  "
                      f"(OOS obs: {n_oos}, min_train: {args.min_train}, "
                      f"step: {args.step})\n")
                _print_oos_table(oos_df)

                for _, row in oos_df.iterrows():
                    all_rows.append({
                        "symbol": symbol,
                        "target": t_label,
                        "horizon": h,
                        **row.to_dict(),
                    })

    if len(symbols) > 1 and all_rows:
        summary = pd.DataFrame(all_rows)

        # append cross-symbol averages for each (target, horizon, model)
        grp_keys = ["target", "horizon", "model"]
        num_cols = ["OOS-R²", "RMSE", "MAE", "bias"]
        grp = summary.groupby(grp_keys, sort=False)
        avg = grp[num_cols].mean().reset_index()
        avg.insert(0, "symbol", "*mean*")
        avg.insert(4, "n_oos", grp["n_oos"].mean().values.astype(int))
        avg = avg[list(summary.columns)]
        full = pd.concat([summary, avg], ignore_index=True)

        print(f"\n{'='*70}")
        print("OOS summary (all symbols)")
        print(f"{'='*70}\n")
        _print_oos_table(full)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntime elapsed (s): {time.time() - t0:.2f}")
