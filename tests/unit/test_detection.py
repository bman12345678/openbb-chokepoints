import numpy as np
import pandas as pd
import pytest

from openbb_chokepoints.core import detection as d


def make(days=365, seed=0, drop=None, level=100.0, noise=3.0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days)
    s = pd.Series(level + rng.normal(0, noise, days), idx)
    if drop:
        a, b, val = drop
        s[a:b] = val
    return s


def test_params_validation():
    with pytest.raises(ValueError):
        d.DetectionParams(k=6, n=5)
    with pytest.raises(ValueError):
        d.DetectionParams(tau=0)
    with pytest.raises(ValueError):
        d.DetectionParams(min_baseline_obs=200)
    with pytest.raises(ValueError):
        d.DetectionParams(merge_gap=-1)
    with pytest.raises(ValueError):
        d.DetectionParams(smooth_days=0)
    assert d.DetectionParams().with_(tau=2.0).tau == 2.0


def test_calm_series_has_no_episode():
    _, eps = d.detect(make())
    assert eps.empty
    assert list(eps.columns) == d.EPISODE_COLUMNS


def test_drop_is_detected_with_short_delay():
    s, eps = d.detect(make(drop=("2022-08-01", "2022-08-20", 40.0)))
    assert len(eps) == 1
    ep = eps.iloc[0]
    assert pd.Timestamp("2022-08-01") <= ep["onset"] <= pd.Timestamp("2022-08-04")
    assert (ep["start"] - pd.Timestamp("2022-08-01")).days <= 5
    assert ep["min_z"] < -3
    assert ep["peak_drop_pct"] < -40
    assert not ep["ongoing"]


def test_single_day_blip_is_ignored():
    s = make()
    s["2022-08-10"] = 0.0
    _, eps = d.detect(s)
    assert eps.empty


def test_spike_up_is_not_flagged():
    s = make(drop=("2022-08-01", "2022-08-20", 200.0))
    _, eps = d.detect(s)
    assert eps.empty


def test_missing_days_stay_nan_and_do_not_crash():
    s = make()
    s["2022-03-01":"2022-03-05"] = np.nan
    scores = d.compute_scores(s)
    assert scores.loc["2022-03-03", "value"] != scores.loc["2022-03-03", "value"]
    assert list(scores.columns) == d.DAILY_COLUMNS


def test_gaps_in_index_are_filled_with_nan():
    s = make().drop(pd.date_range("2022-05-01", "2022-05-03"))
    scores = d.compute_scores(s)
    assert len(scores) == 365
    assert scores.loc["2022-05-02", "value"] != scores.loc["2022-05-02", "value"]


def test_warmup_has_no_flags():
    scores = d.compute_scores(make(days=30, drop=("2022-01-10", "2022-01-30", 10.0)))
    assert scores["z"].isna().all()
    assert not scores["flag"].any()


def test_baseline_lag_blocks_self_contamination():
    s = make(days=400, drop=("2022-06-01", "2022-12-31", 40.0))
    scores = d.compute_scores(s)
    # three weeks into the long outage it is still abnormal
    assert scores.loc["2022-06-22", "z"] < -3


def test_scale_floor_prevents_division_blowup():
    s = pd.Series(50.0, pd.date_range("2022-01-01", periods=200))
    s.iloc[150:] = 49.0  # tiny change on a perfectly flat series
    scores = d.compute_scores(s)
    assert scores["z"].abs().max() < 3
    assert scores["baseline_scale"].dropna().min() >= 1.0


def test_episode_merge_gap():
    s = make(days=300, drop=("2022-08-01", "2022-08-20", 40.0))
    s["2022-09-01":"2022-09-20"] = 40.0  # second outage after ~12 normal days
    p = d.DetectionParams(smooth_days=1)
    _, eps = d.detect(s, p)
    assert len(eps) == 2
    _, merged = d.detect(s, p.with_(merge_gap=20))
    assert len(merged) == 1


def test_ongoing_flag():
    s = make(drop=("2022-12-10", "2022-12-31", 30.0))
    _, eps = d.detect(s)
    assert eps.iloc[-1]["ongoing"]


def test_yoy_uses_364_day_shift():
    idx = pd.date_range("2021-01-01", periods=800)
    s = pd.Series(100.0, idx)
    s["2022-01-01":] = 50.0
    scores = d.compute_scores(s, d.DetectionParams(smooth_days=1))
    assert scores["yoy"].loc["2022-06-01"] == pytest.approx(-0.5)


def test_apply_rule_matches_full_run():
    s = make(drop=("2022-08-01", "2022-08-20", 40.0))
    base = d.compute_scores(s)
    p2 = d.DetectionParams(tau=2.5, k=2, n=3)
    assert d.apply_rule(base, p2)["flag"].equals(d.compute_scores(s, p2)["flag"])


def test_input_validation_and_empty():
    with pytest.raises(TypeError):
        d.compute_scores(pd.Series([1.0, 2.0]))
    assert d.compute_scores(
        pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    ).empty


def test_duplicate_dates_keep_last():
    s = make(days=200)
    dup = pd.concat([s, s.iloc[[10]] * 0 + 5.0])
    scores = d.compute_scores(dup)
    assert len(scores) == 200
    assert scores["value"].iloc[10] == 5.0


def _volume(chokepoints):
    parts = []
    for i, (name, drop) in enumerate(chokepoints.items()):
        s = make(seed=i, drop=drop)
        parts.append(
            pd.DataFrame(
                {
                    "date": s.index,
                    "chokepoint": name,
                    "vessel_type": "total",
                    "vessels": s.values,
                    "capacity_tons": np.nan,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_detect_all():
    vol = _volume({"a": ("2022-08-01", "2022-08-20", 40.0), "b": None})
    eps = d.detect_all(vol)
    assert list(eps["chokepoint"].unique()) == ["a"]
    assert d.detect_all(vol[vol["chokepoint"] == "b"]).empty
    with pytest.raises(ValueError):
        d.detect_all(vol.drop(columns="vessels"))


def test_current_status():
    s = make(drop=("2022-12-10", "2022-12-31", 30.0))
    st = d.current_status(s)
    assert st["flag"] is True
    assert st["z"] < -3
    assert st["date"] == s.index[-1]
    empty = d.current_status(
        pd.Series([np.nan] * 10, pd.date_range("2022-01-01", periods=10))
    )
    assert empty["date"] is None
