"""Unit tests for the market-model event study (synthetic data, no network)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from openbb_chokepoints.core.event_study import (
    EventStudyWarning,
    abnormal_returns,
    car_and_test,
    fit_market_model,
    format_window,
    run_event_study,
)
from openbb_chokepoints.data.market import (
    get_returns,
    history_range,
    long_to_wide,
    prices_to_returns,
)

ALPHA, BETA, SIGMA_EPS = 0.0002, 1.2, 0.015
MU_M, SIGMA_M = 0.0004, 0.01
N_DAYS = 300
DAY0_POS = 260  # leaves 260 days before day 0 and 39 after
DATES = pd.bdate_range("2022-01-03", periods=N_DAYS)
EVENT_DATE = DATES[DAY0_POS]


def simulate(rng: np.random.Generator, shock: float = 0.0):
    """One asset following the market model, with ``shock`` added on day 0."""
    r_m = rng.normal(MU_M, SIGMA_M, N_DAYS)
    r_i = ALPHA + BETA * r_m + rng.normal(0.0, SIGMA_EPS, N_DAYS)
    r_i[DAY0_POS] += shock
    return (
        pd.DataFrame({"SIM": r_i}, index=DATES),
        pd.Series(r_m, index=DATES, name="MKT"),
    )


# --- OLS -----------------------------------------------------------------


def test_ols_matches_statsmodels():
    rng = np.random.default_rng(0)
    r_m = rng.normal(MU_M, SIGMA_M, 221)
    r_i = ALPHA + BETA * r_m + rng.normal(0.0, SIGMA_EPS, 221)

    alpha, beta, sigma2, XtX_inv, n_est = fit_market_model(r_i, r_m)
    fit = sm.OLS(r_i, sm.add_constant(r_m)).fit()

    assert n_est == 221
    assert alpha == pytest.approx(fit.params[0], abs=1e-10)
    assert beta == pytest.approx(fit.params[1], abs=1e-10)
    assert sigma2 == pytest.approx(fit.scale, abs=1e-10)
    np.testing.assert_allclose(XtX_inv, fit.normalized_cov_params, rtol=1e-8)


def test_ols_aligns_series_and_drops_missing():
    rng = np.random.default_rng(1)
    asset, market = simulate(rng)
    r_i = asset["SIM"].copy()
    r_i.iloc[[3, 10]] = np.nan
    shuffled_market = market.sample(frac=1.0, random_state=1)

    alpha, beta, sigma2, _, n_est = fit_market_model(r_i, shuffled_market)
    keep = r_i.notna()
    fit = sm.OLS(r_i[keep], sm.add_constant(market[keep])).fit()

    assert n_est == N_DAYS - 2
    assert alpha == pytest.approx(fit.params.iloc[0], abs=1e-10)
    assert beta == pytest.approx(fit.params.iloc[1], abs=1e-10)
    assert sigma2 == pytest.approx(fit.scale, abs=1e-10)


def test_ols_rejects_too_few_rows():
    with pytest.raises(ValueError):
        fit_market_model(np.array([0.01, 0.02]), np.array([0.0, 0.01]))


# --- AR / CAR ------------------------------------------------------------


def test_abnormal_returns_formula():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r_i = pd.Series([0.01, 0.02, -0.01], index=idx)
    r_m = pd.Series([0.005, np.nan, 0.0], index=idx)
    ar = abnormal_returns(r_i, r_m, alpha=0.001, beta=2.0)
    assert ar.iloc[0] == pytest.approx(0.01 - (0.001 + 2.0 * 0.005))
    assert np.isnan(ar.iloc[1])
    assert ar.iloc[2] == pytest.approx(-0.011)


def test_car_variance_matches_matrix_form():
    """sigma2 (L2 + g'(X'X)^-1 g) equals 1' V 1 with V = sigma2 (I + X*(X'X)^-1 X*')."""
    rng = np.random.default_rng(2)
    r_m_est = rng.normal(MU_M, SIGMA_M, 200)
    r_i_est = ALPHA + BETA * r_m_est + rng.normal(0.0, SIGMA_EPS, 200)
    alpha, beta, sigma2, XtX_inv, _ = fit_market_model(r_i_est, r_m_est)

    r_m_evt = rng.normal(MU_M, SIGMA_M, 11)
    ar = rng.normal(0.0, SIGMA_EPS, 11)
    car, var, t, p = car_and_test(ar, r_m_evt, sigma2, XtX_inv)

    X_star = np.column_stack([np.ones(11), r_m_evt])
    V = sigma2 * (np.eye(11) + X_star @ XtX_inv @ X_star.T)
    ones = np.ones(11)
    assert car == pytest.approx(ar.sum())
    assert var == pytest.approx(ones @ V @ ones, rel=1e-12)
    assert var > 11 * sigma2  # estimation error only adds variance
    assert t == pytest.approx(car / np.sqrt(var))
    assert 0.0 <= p <= 1.0


def test_car_zero_variance_gives_nan_test():
    car, var, t, p = car_and_test(
        np.zeros(3), np.zeros(3), sigma2=0.0, XtX_inv=np.eye(2)
    )
    assert car == 0.0 and var == 0.0
    assert np.isnan(t) and np.isnan(p)


def test_car_empty_window_raises():
    with pytest.raises(ValueError):
        car_and_test(np.array([np.nan]), np.array([0.0]), 1e-4, np.eye(2))


# --- Monte Carlo: recovery and size --------------------------------------


def test_recovery_of_injected_abnormal_return():
    rng = np.random.default_rng(42)
    n_sims, shock = 2000, 0.05
    day0_ar, rejections = np.empty(n_sims), 0
    for k in range(n_sims):
        asset, market = simulate(rng, shock=shock)
        rows = run_event_study(asset, market, EVENT_DATE, sub_windows=[(0, 0)])
        one_day = next(r for r in rows if r["window"] == "(0, 0)")
        day0_ar[k] = one_day["ar_by_day"][5]  # (-5, 20): index 5 is day 0
        assert one_day["car"] == pytest.approx(day0_ar[k])
        rejections += one_day["p_value"] < 0.05

    assert abs(day0_ar.mean() - shock) < 0.001
    assert rejections / n_sims > 0.85


def test_size_of_full_window_car_test():
    rng = np.random.default_rng(7)
    n_sims, rejections = 4000, 0
    for _ in range(n_sims):
        asset, market = simulate(rng)
        rows = run_event_study(asset, market, EVENT_DATE, sub_windows=())
        assert len(rows) == 1 and rows[0]["window"] == "(-5, 20)"
        rejections += rows[0]["p_value"] < 0.05

    assert 0.035 <= rejections / n_sims <= 0.065


# --- Day-0 alignment and windows ------------------------------------------


def test_saturday_event_maps_to_next_trading_day():
    rng = np.random.default_rng(3)
    asset, market = simulate(rng)
    saturday = pd.Timestamp("2022-12-03")
    assert saturday.day_of_week == 5

    rows = run_event_study(asset, market, saturday.date())
    assert rows[0]["event_date"] == date(2022, 12, 3)
    assert rows[0]["day0"] == date(2022, 12, 5)  # Monday


def test_day0_uses_each_symbols_own_index():
    rng = np.random.default_rng(4)
    asset, market = simulate(rng)
    monday = pd.Timestamp("2022-12-05")
    asset["HOL"] = asset["SIM"]
    asset.loc[monday, "HOL"] = np.nan  # HOL's exchange is closed on Monday

    rows = run_event_study(asset, market, "2022-12-03")
    day0 = {r["symbol"]: r["day0"] for r in rows}
    assert day0 == {"SIM": date(2022, 12, 5), "HOL": date(2022, 12, 6)}


def test_trading_day_offsets_and_ar_by_day():
    rng = np.random.default_rng(5)
    asset, market = simulate(rng)
    rows = run_event_study(asset, market, EVENT_DATE, est_window=(-250, -30))
    full = rows[0]
    r_i, r_m = asset["SIM"], market

    est = slice(DAY0_POS - 250, DAY0_POS - 30 + 1)
    alpha, beta, *_ = fit_market_model(r_i.iloc[est], r_m.iloc[est])
    expected_ar = r_i - (alpha + beta * r_m)

    assert full["n_est"] == 221
    assert full["alpha"] == pytest.approx(alpha)
    assert full["beta"] == pytest.approx(beta)
    assert len(full["ar_by_day"]) == 26
    np.testing.assert_allclose(
        full["ar_by_day"], expected_ar.iloc[DAY0_POS - 5 : DAY0_POS + 21]
    )
    assert full["car"] == pytest.approx(
        expected_ar.iloc[DAY0_POS - 5 : DAY0_POS + 21].sum()
    )


def test_reports_full_window_then_fitting_sub_windows():
    rng = np.random.default_rng(6)
    asset, market = simulate(rng)
    rows = run_event_study(asset, market, EVENT_DATE)
    assert [r["window"] for r in rows] == ["(-5, 20)", "(-1, 1)", "(0, 5)", "(0, 20)"]

    rows = run_event_study(asset, market, EVENT_DATE, evt_window=(-2, 10))
    assert [r["window"] for r in rows] == ["(-2, 10)", "(-1, 1)", "(0, 5)"]
    assert format_window((-2, 10)) == "(-2, 10)"


def test_truncated_event_window_warns():
    rng = np.random.default_rng(8)
    asset, market = simulate(rng)
    late_event = DATES[N_DAYS - 3]
    with pytest.warns(EventStudyWarning, match="truncated, 8 of 26"):
        rows = run_event_study(asset, market, late_event)
    full = rows[0]
    assert full["ar_by_day"][8:] == [None] * 18
    assert full["car"] == pytest.approx(sum(full["ar_by_day"][:8]))


@pytest.mark.parametrize(
    ("est_window", "evt_window", "min_obs"),
    [
        ((-30, -250), (-5, 20), 120),
        ((-250, -5), (-5, 20), 120),
        ((-250, -30), (5, -5), 120),
        ((-250, -30), (-5, 20), 2),
    ],
)
def test_invalid_windows_raise(est_window, evt_window, min_obs):
    rng = np.random.default_rng(9)
    asset, market = simulate(rng)
    with pytest.raises(ValueError):
        run_event_study(asset, market, EVENT_DATE, est_window, evt_window, min_obs)


# --- Missing data -----------------------------------------------------------


def test_too_little_history_skips_symbol_with_warning():
    rng = np.random.default_rng(10)
    asset, market = simulate(rng)
    asset["NEW"] = asset["SIM"]
    asset.iloc[: DAY0_POS - 100, asset.columns.get_loc("NEW")] = np.nan  # 71 est. days

    with pytest.warns(EventStudyWarning, match="NEW: only 71 estimation"):
        rows = run_event_study(asset, market, EVENT_DATE)
    assert {r["symbol"] for r in rows} == {"SIM"}


def test_event_after_last_trading_day_skips_symbol():
    rng = np.random.default_rng(11)
    asset, market = simulate(rng)
    with pytest.warns(EventStudyWarning, match="no trading day on or after"):
        assert run_event_study(asset, market, DATES[-1] + pd.Timedelta(days=3)) == []


def test_missing_market_days_are_dropped_not_zeroed():
    rng = np.random.default_rng(12)
    asset, market = simulate(rng)
    market_gap = market.drop(DATES[DAY0_POS - 100])  # inside the estimation window
    market_gap = market_gap.drop(DATES[DAY0_POS + 2])  # inside the event window

    rows = run_event_study(asset, market_gap, EVENT_DATE)
    full = rows[0]
    assert full["n_est"] == 220
    assert full["ar_by_day"][7] is None
    assert full["car"] == pytest.approx(
        sum(v for v in full["ar_by_day"] if v is not None)
    )


def test_prices_to_returns_does_not_forward_fill():
    idx = pd.bdate_range("2024-07-01", periods=5)  # Mon 1 .. Fri 5 July
    prices = pd.DataFrame(
        {
            "US": [100.0, 101.0, 102.0, np.nan, 104.0],
            "DK": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=idx,
    )
    returns = prices_to_returns(prices)
    july4 = pd.Timestamp("2024-07-04")

    assert np.isnan(returns.loc[july4, "US"])  # the missing day stays missing
    assert (returns["US"].dropna() != 0.0).all()  # no fake 0% return
    assert returns.loc["2024-07-05", "US"] == pytest.approx(104.0 / 102.0 - 1)
    assert returns.loc[july4, "DK"] == pytest.approx(53.0 / 52.0 - 1)
    assert returns["US"].notna().sum() == 3


def test_get_returns_pipeline_without_network(monkeypatch):
    import openbb_chokepoints.data.market as market

    idx = pd.bdate_range("2024-07-01", periods=5)
    long = pd.DataFrame(
        {
            "date": list(idx) + [d for d in idx if d.day != 4],
            "symbol": ["DK"] * 5 + ["US"] * 4,
            "close": [50.0, 51.0, 52.0, 53.0, 54.0, 100.0, 101.0, 102.0, 104.0],
        }
    ).set_index("date")
    monkeypatch.setattr(market, "fetch_prices", lambda s, a, b: long_to_wide(long))

    returns = get_returns(["US", "DK"], date(2024, 7, 1), date(2024, 7, 5))
    assert np.isnan(returns.loc["2024-07-04", "US"])
    assert returns.loc["2024-07-05", "US"] == pytest.approx(104.0 / 102.0 - 1)


def test_history_range_covers_windows():
    start, end = history_range(
        date(2023, 12, 15), (-250, -30), (-5, 20), today=date(2026, 1, 1)
    )
    assert 1.5 < (date(2023, 12, 15) - start).days / 365.25 < 1.7
    assert (end - date(2023, 12, 15)).days >= 28

    _, end = history_range(
        date(2026, 1, 1), (-250, -30), (-5, 20), today=date(2026, 1, 2)
    )
    assert end == date(2026, 1, 2)


def test_sub_window_without_data_is_skipped_with_warning():
    rng = np.random.default_rng(13)
    asset, market = simulate(rng)
    with pytest.warns(EventStudyWarning) as record:
        rows = run_event_study(asset, market, DATES[N_DAYS - 3], sub_windows=[(10, 15)])
    messages = [str(w.message) for w in record]
    assert any("truncated, 8 of 26" in m for m in messages)
    assert any("no data in CAR window (10, 15)" in m for m in messages)
    assert [r["window"] for r in rows] == ["(-5, 20)"]


def test_timezone_aware_and_unsorted_index():
    rng = np.random.default_rng(14)
    asset, market = simulate(rng)
    expected = run_event_study(asset, market, EVENT_DATE)

    asset_tz = asset.tz_localize("America/New_York").iloc[::-1]
    market_tz = market.tz_localize("America/New_York")
    rows = run_event_study(asset_tz, market_tz, EVENT_DATE)
    assert rows[0]["day0"] == expected[0]["day0"]
    assert rows[0]["car"] == pytest.approx(expected[0]["car"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        fit_market_model(np.zeros(5), np.zeros(4))
