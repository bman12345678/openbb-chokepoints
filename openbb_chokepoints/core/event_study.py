"""Market-model event study for a single asset.

Method: MacKinlay (1997), "Event Studies in Economics and Finance", and
Campbell, Lo & MacKinlay (1997), The Econometrics of Financial Markets, ch. 4.

For each asset i, with day 0 the first trading day on or after the event date
(counted in the asset's own trading calendar):

1. Fit R_it = alpha_i + beta_i * R_mt + e_it by OLS over the estimation window.
   sigma2_i = sum(e_hat^2) / (L1 - 2).
2. AR_it = R_it - (alpha_hat_i + beta_hat_i * R_mt) over the event window.
3. CAR_i(t1, t2) = sum of AR_it over the window.
4. Var[CAR_i] = sigma2_i * (L2 + g' (X'X)^-1 g), where L2 is the number of days
   in the CAR window and g holds the column sums of [1, R_m] over that window.
   This includes the estimation error in alpha_hat and beta_hat.
5. t_i = CAR_i / sqrt(Var[CAR_i]); two-sided p-value from the standard normal.

Pure NumPy/pandas/SciPy. Returns are simple returns stored as decimals.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

Window = tuple[int, int] # represents inclusive relative day offsets (x days before event, y days after event)
#global defaults, subject to change
DEFAULT_ESTIMATION_WINDOW: Window = (-250, -30)
DEFAULT_EVENT_WINDOW: Window = (-5, 20)
DEFAULT_MIN_ESTIMATION_OBS = 120
DEFAULT_SUB_WINDOWS: tuple[Window, ...] = ((-1, 1), (0, 5), (0, 20))

#subclass that notifies when asset is skipped because something due to not enough / wrong data 
class EventStudyWarning(UserWarning):
    """A data problem that skipped a symbol or shortened its event window."""

#helper function to convert integer window into string label
def format_window(window: Window) -> str:
    """Label a window the way it is reported, e.g. ``"(-5, 20)"``."""
    return f"({window[0]}, {window[1]})"

#function for fitting OLS regression over estimation window
def fit_market_model(
    r_asset: pd.Series | np.ndarray, r_market: pd.Series | np.ndarray
) -> tuple[float, float, float, np.ndarray, int]:
    """Fit the market model by OLS.

    Series are aligned on their index; arrays must have the same length.
    Rows where either return is missing are dropped.

    Returns
    -------
    (alpha, beta, sigma2, XtX_inv, n_est)
        sigma2 is the residual variance with L1 - 2 degrees of freedom and
        XtX_inv is the pseudo-inverse of X'X for X = [1, R_m].
    """
   #helper to strip missing values
    y, x = _aligned_arrays(r_asset, r_market)
   #checking degrees of freedom
    n_est = len(y)
    if n_est < 3:
        raise ValueError(f"Need at least 3 estimation observations, got {n_est}.")
#constructing design matrix (row for each trading day, one column of 1s and one column of daily market return)
#used to represent y = mx + b, (market's daily returns, asset's daily returns), m = stock's sensitivty to market
    X = np.column_stack([np.ones(n_est), x])
   #solves the ordinary least squares equation to develop the linear model
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
   #residual
    resid = y - X @ coef
   #variance
    sigma2 = float(resid @ resid / (n_est - 2))
   #computes inverse of transpose of X * X (Moore-Penrose pseudo-inverse to account for unsual data)
   #covariance matrix needed for Abnormal Returns later on
    XtX_inv = np.linalg.pinv(X.T @ X)
    return float(coef[0]), float(coef[1]), sigma2, XtX_inv, n_est


def abnormal_returns(
    r_asset: pd.Series, r_market: pd.Series, alpha: float, beta: float
) -> pd.Series:
    """AR_t = R_t - (alpha + beta * R_mt), on the asset's index.

    Days where the market return is missing give a missing AR.
    """
   #a.k.a. observed - expected (residual for out of estimation window / points not in sample used to make model)
    rm = r_market.reindex(r_asset.index)
    return (r_asset - (alpha + beta * rm)).rename(r_asset.name)


def car_and_test(
    ar: pd.Series | np.ndarray,
    r_market_window: pd.Series | np.ndarray,
    sigma2: float,
    XtX_inv: np.ndarray,
) -> tuple[float, float, float, float]:
    """CAR over a window and its t-test with estimation error included.

    ``ar`` and ``r_market_window`` cover the same CAR window. Days where either
    is missing are dropped, and L2 counts the remaining days.

    Returns
    -------
    (car, var, t, p)
        t and p are NaN when the variance is not positive.
    """
    a, m = _aligned_arrays(ar, r_market_window)
    l2 = len(a)
    if l2 == 0:
        raise ValueError("CAR window has no observations.")
   #cumulative abnormal returns
    car = float(a.sum())
   #vector (2x1) cumulative weight of regressors on window
    g = np.array([l2, m.sum()])
    var = float(sigma2 * (l2 + g @ XtX_inv @ g))
    if not var > 0:
        return car, var, float("nan"), float("nan")
   #t statistic
    t = car / np.sqrt(var)
   #two tailed p value (abnormal reaction can be in either direction)
    p = float(2 * norm.sf(abs(t)))
    return car, var, float(t), p


def run_event_study(
    returns_df: pd.DataFrame,
    market_series: pd.Series,
    event_date: date | str | pd.Timestamp,
    est_window: Window = DEFAULT_ESTIMATION_WINDOW,
    evt_window: Window = DEFAULT_EVENT_WINDOW,
    min_obs: int = DEFAULT_MIN_ESTIMATION_OBS,
    sub_windows: Sequence[Window] = DEFAULT_SUB_WINDOWS,
) -> list[dict[str, Any]]:
    """Run the market-model event study for every column of ``returns_df``.

    Parameters
    ----------
    returns_df
        Simple returns, one column per symbol, indexed by date. A missing value
        means the symbol did not trade that day; it is never treated as 0%.
    market_series
        Simple returns of the market index, indexed by date.
    event_date
        Calendar date of the event. Day 0 is the first date in each symbol's
        own (non-missing) return index on or after this date.
    est_window, evt_window
        Inclusive (start, end) trading-day offsets from day 0. The estimation
        window must end before the event window starts.
    min_obs
        Symbols with fewer estimation rows (after dropping days without a
        market return) are skipped with an ``EventStudyWarning``.
    sub_windows
        Extra CAR windows reported when they fit inside ``evt_window``.

    Returns
    -------
    list of dict
        One row per symbol and CAR window, the full event window first.
    """
    _validate_windows(est_window, evt_window, min_obs)

    event_ts = pd.Timestamp(event_date).normalize()
    market = _clean_index(market_series)
    windows = [evt_window] + [
        w
        for w in dict.fromkeys(tuple(w) for w in sub_windows)
        if evt_window[0] <= w[0] <= w[1] <= evt_window[1] and w != tuple(evt_window)
    ]

    rows: list[dict[str, Any]] = []
    for symbol in returns_df.columns:
        rows.extend(
            _study_one(
                symbol=str(symbol),
                asset=_clean_index(returns_df[symbol]).dropna(),
                market=market,
                event_ts=event_ts,
                est_window=est_window,
                evt_window=evt_window,
                windows=windows,
                min_obs=min_obs,
            )
        )
    return rows


def _study_one(
    symbol: str,
    asset: pd.Series,
    market: pd.Series,
    event_ts: pd.Timestamp,
    est_window: Window,
    evt_window: Window,
    windows: list[Window],
    min_obs: int,
) -> list[dict[str, Any]]:
    idx = asset.index
    pos0 = int(idx.searchsorted(event_ts, side="left"))
    if pos0 >= len(idx):
        _warn(f"{symbol}: no trading day on or after {event_ts.date()}; skipped.")
        return []
    day0 = idx[pos0]

    est_start = max(pos0 + est_window[0], 0)
    est_end = pos0 + est_window[1]
    est = asset.iloc[est_start : est_end + 1] if est_end >= 0 else asset.iloc[:0]
    r_m_est = market.reindex(est.index)
    n_usable = int((est.notna() & r_m_est.notna()).sum())
    if n_usable < max(min_obs, 3):
        _warn(
            f"{symbol}: only {n_usable} estimation observations "
            f"(min_estimation_obs={min_obs}); skipped."
        )
        return []
    alpha, beta, sigma2, XtX_inv, n_est = fit_market_model(est, r_m_est)

    # AR and R_m by relative day over the full event window; NaN where the
    # day is beyond the data or the market return is missing.
    rel_days = np.arange(evt_window[0], evt_window[1] + 1)
    ar = np.full(len(rel_days), np.nan)
    r_m = np.full(len(rel_days), np.nan)
    positions = pos0 + rel_days
    in_range = (positions >= 0) & (positions < len(idx))
    dates = idx[positions[in_range]]
    r_m[in_range] = market.reindex(dates).to_numpy(dtype=float)
    ar[in_range] = abnormal_returns(asset.loc[dates], market, alpha, beta).to_numpy()

    n_available = int(in_range.sum())
    if n_available < len(rel_days):
        _warn(
            f"{symbol}: event window {format_window(evt_window)} truncated, "
            f"{n_available} of {len(rel_days)} trading days available."
        )

    common = {
        "symbol": symbol,
        "event_date": event_ts.date(),
        "day0": day0.date(),
        "alpha": alpha,
        "beta": beta,
        "sigma_eps": float(np.sqrt(sigma2)),
        "n_est": n_est,
        "ar_by_day": [None if np.isnan(v) else float(v) for v in ar],
    }

    rows = []
    for window in windows:
        sel = (rel_days >= window[0]) & (rel_days <= window[1])
        if not np.isfinite(ar[sel]).any():
            _warn(f"{symbol}: no data in CAR window {format_window(window)}.")
            continue
        car, _, t, p = car_and_test(ar[sel], r_m[sel], sigma2, XtX_inv)
        rows.append(
            {
                **common,
                "window": format_window(window),
                "car": car,
                "car_t": None if np.isnan(t) else t,
                "p_value": None if np.isnan(p) else p,
            }
        )
    return rows


def _aligned_arrays(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, join="inner")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Length mismatch: {x.shape} vs {y.shape}.")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _clean_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index().astype(float)


def _validate_windows(est_window: Window, evt_window: Window, min_obs: int) -> None:
    if not est_window[0] <= est_window[1]:
        raise ValueError(f"Invalid estimation window {est_window}.")
    if not evt_window[0] <= evt_window[1]:
        raise ValueError(f"Invalid event window {evt_window}.")
    if not est_window[1] < evt_window[0]:
        raise ValueError(
            f"Estimation window {est_window} must end before the event window "
            f"{evt_window} starts."
        )
    if min_obs < 3:
        raise ValueError("min_obs must be at least 3.")


def _warn(message: str) -> None:
    warnings.warn(message, EventStudyWarning, stacklevel=3)
