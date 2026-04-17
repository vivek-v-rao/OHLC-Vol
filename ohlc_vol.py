"""
ohlc_vol.py

Compute OHLC-based annualised volatility estimators from daily price data.

Volatility estimators (all annualised, × sqrt(252)):
  close-to-close        |log(C_t / C_{t-1})|  * sqrt(252)
  close-to-open         |log(O_t / C_{t-1})|  * sqrt(252)   overnight
  open-to-close         |log(C_t / O_t)|       * sqrt(252)   daytime
  combined co+oc        sqrt((r_co^2 + r_oc^2) * 252)
  Parkinson             sqrt(log(H/L)^2 / (4 ln 2)  * 252)
  Garman-Klass          sqrt(max(0, 0.5 log(H/L)^2 - (2ln2-1) log(C/O)^2) * 252)
  Rogers-Satchell       sqrt(max(0, log(H/O)log(H/C) + log(L/O)log(L/C)) * 252)
  Yang-Zhang (rolling)  sqrt(252 * (var_o + k*var_c + (1-k)*var_rs))
  True Range            sqrt(log(max(H,C_prev)/min(L,C_prev))^2 / (4 ln 2) * 252)
  overnight + RS        sqrt((log(O/C_prev)^2 + RS_intraday) * 252)
  adj close-to-close    close-to-close on split/dividend-adjusted close price
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────
TRADING_DAYS = 252.0
RETURN_SCALING = 100.0      # returns stored as percent
YZ_WINDOW = 21              # rolling window for Yang-Zhang estimator
# Set False to drop unadjusted close-to-close (redundant when adj-close-to-close
# is present; differs only on ex-dividend / split days).  Set True to restore it.
INCLUDE_UNADJ_CC = False

VOL_LABELS: dict[str, str] = {
    "vol_cc_ann":        "close-to-close",
    "vol_adj_cc_ann":    "adj-close-to-close",
    "vol_co_ann":        "close-to-open",
    "vol_oc_ann":        "open-to-close",
    "vol_co_oc_ann":     "co+oc-combined",
    "vol_parkinson_ann": "parkinson",
    "vol_gk_ann":        "garman-klass",
    "vol_rs_ann":        "rogers-satchell",
    "vol_yz_ann":        "yang-zhang",
    "vol_tr_ann":        "true-range",
    "vol_on_rs_ann":     "overnight+rs",
    "vol_meilijson_ann": "meilijson",
    "vol_ht_ann":        "hodges-tompkins",
    "neg_ret_adj_cc":     "neg-adj-cc-ret",
    "neg_ret_adj_cc_pos": "neg-adj-cc-ret+",
}

_ALL_VOL_COLS: list[str] = list(VOL_LABELS.keys())
OHLC_VOL_COLS: list[str] = [
    c for c in _ALL_VOL_COLS if INCLUDE_UNADJ_CC or c != "vol_cc_ann"
]


def clean_label(col: str) -> str:
    return VOL_LABELS.get(col, col)


def compute_vol_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all OHLC-based annualised volatility measures.

    All intermediate return series are stored as percent (× RETURN_SCALING)
    so that variances are in percent² and annualised vols in percent × √252.
    """
    out = df.copy()
    s = RETURN_SCALING

    o = out["Open"]
    h = out["High"]
    l = out["Low"]
    c = out["Close"]
    ac = out["Adj_Close"]
    c_prev = c.shift(1)

    # ── return series ──────────────────────────────────────────────────────────
    out["ret_cc"]     = s * np.log(c  / c_prev)          # close-to-close
    out["ret_co"]     = s * np.log(o  / c_prev)          # overnight (close→open)
    out["ret_oc"]     = s * np.log(c  / o)               # daytime   (open→close)
    out["ret_adj_cc"] = s * np.log(ac / ac.shift(1))     # adj close-to-close
    out["neg_ret_adj_cc"]     = -out["ret_adj_cc"]                        # leverage-effect predictor
    out["neg_ret_adj_cc_pos"] = out["neg_ret_adj_cc"].clip(lower=0.0)   # positive part only (down days)

    # ── scalar vol measures ────────────────────────────────────────────────────
    ann = np.sqrt(TRADING_DAYS)

    out["vol_cc_ann"]     = ann * out["ret_cc"].abs()
    out["vol_adj_cc_ann"] = ann * out["ret_adj_cc"].abs()
    out["vol_co_ann"]     = ann * out["ret_co"].abs()
    out["vol_oc_ann"]     = ann * out["ret_oc"].abs()
    out["vol_co_oc_ann"]  = np.sqrt(
        TRADING_DAYS * (out["ret_co"].pow(2) + out["ret_oc"].pow(2))
    )

    # Parkinson (1980)
    hl = s * np.log(h / l)
    out["vol_parkinson_ann"] = np.sqrt(
        (TRADING_DAYS * hl.pow(2) / (4.0 * np.log(2.0))).clip(lower=0.0)
    )

    # Garman-Klass (1980)
    oc_log = s * np.log(c / o)
    var_gk = TRADING_DAYS * (
        0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * oc_log.pow(2)
    )
    out["vol_gk_ann"] = np.sqrt(var_gk.clip(lower=0.0))

    # Rogers-Satchell (1991)
    lho = np.log(h / o)
    lhc = np.log(h / c)
    llo = np.log(l / o)
    llc = np.log(l / c)
    rs_var = RETURN_SCALING**2 * (lho * lhc + llo * llc)
    out["vol_rs_ann"] = np.sqrt((TRADING_DAYS * rs_var).clip(lower=0.0))

    # Yang-Zhang (2000) — rolling window
    n_yz = YZ_WINDOW
    k_yz = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1))
    var_o  = out["ret_co"].rolling(n_yz).var()
    var_c  = out["ret_oc"].rolling(n_yz).var()
    var_rs = rs_var.rolling(n_yz).mean()
    var_yz = (var_o + k_yz * var_c + (1.0 - k_yz) * var_rs).clip(lower=0.0)
    out["vol_yz_ann"] = np.sqrt(TRADING_DAYS * var_yz)

    # True Range (Wilder 1978) — Parkinson extended to include overnight gaps
    # True high = max(H_t, C_{t-1}),  true low = min(L_t, C_{t-1})
    # Same Parkinson formula applied to the extended range.
    tr_high = pd.concat([h, c_prev], axis=1).max(axis=1)
    tr_low  = pd.concat([l, c_prev], axis=1).min(axis=1)
    tr_hl   = s * np.log(tr_high / tr_low)
    out["vol_tr_ann"] = np.sqrt(
        (TRADING_DAYS * tr_hl.pow(2) / (4.0 * np.log(2.0))).clip(lower=0.0)
    )

    # Overnight + Rogers-Satchell — single-period decomposition of Yang-Zhang
    # σ² = var_overnight + var_RS_intraday
    # var_overnight = log(O/C_prev)²  (no mean subtraction; daily drift ≈ 0)
    # var_RS_intraday = RS estimator (already computed as rs_var above, in %²)
    on_var  = out["ret_co"].pow(2)          # overnight variance, in %²
    out["vol_on_rs_ann"] = np.sqrt(
        (TRADING_DAYS * (on_var + rs_var)).clip(lower=0.0)
    )

    # Meilijson (2009) — minimum-variance quadratic unbiased estimator
    # Uses S2 sufficient statistic: flip the intraday path on down days so
    # the close is always above the open, preserving all information about σ².
    # Four basic unbiased estimators (eq. 1) are combined with optimal weights
    # (eq. 3) that minimise variance (efficiency 7.73 vs 7.40 for Garman-Klass).
    c_oc = s * np.log(c / o)        # log(Close/Open), intraday return
    h_o  = s * np.log(h / o)        # log(High/Open)  ≥ 0
    l_o  = s * np.log(l / o)        # log(Low/Open)   ≤ 0
    pos  = c_oc >= 0                 # True on up days
    C_s2 = c_oc.abs()               # always ≥ 0
    H_s2 = h_o.where(pos, -l_o)     # always ≥ 0
    L_s2 = l_o.where(pos, -h_o)     # always ≤ 0
    _d4  = 2.0 * np.log(2.0) - 1.25  # ≈ 0.1363, denominator of σ̂²₄
    m1 = 2.0 * ((H_s2 - C_s2)**2 + L_s2**2)
    m2 = C_s2**2
    m3 = 2.0 * (H_s2 - C_s2 - L_s2) * C_s2
    m4 = -(H_s2 - C_s2) * L_s2 / _d4
    var_mei = 0.273520*m1 + 0.160358*m2 + 0.365212*m3 + 0.200910*m4
    out["vol_meilijson_ann"] = np.sqrt((TRADING_DAYS * var_mei).clip(lower=0.0))

    # Hodges-Tompkins (2002) — rolling close-to-close with overlapping-data bias correction
    # Eq. (12): adjustment = 1 / (1 - h/n + h²/(3n²)), where n = T - h + 1
    n_ht  = YZ_WINDOW
    T_ht  = len(out)
    n_obs = max(T_ht - n_ht + 1, 1)
    adj_ht = 1.0 / (1.0 - n_ht / n_obs + n_ht**2 / (3.0 * n_obs**2))
    var_ht = out["ret_adj_cc"].pow(2).rolling(n_ht).mean() * adj_ht
    out["vol_ht_ann"] = np.sqrt((TRADING_DAYS * var_ht).clip(lower=0.0))

    return out
