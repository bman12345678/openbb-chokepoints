import numpy as np
import pandas as pd
import pytest

from openbb_chokepoints.core import detection as d
from openbb_chokepoints.core import validation as v


def ts(x):
    return pd.Timestamp(x)


def events(*rows):
    cols = ["id", "type", "chokepoint", "related_chokepoints", "news_date", "end_date"]
    out = pd.DataFrame(rows, columns=cols)
    out["news_date"] = pd.to_datetime(out["news_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    return out


def episodes(*rows):
    df = pd.DataFrame(rows, columns=["chokepoint", "onset", "start", "end"])
    for c in ("onset", "start", "end"):
        df[c] = pd.to_datetime(df[c])
    return df


EV = events(
    ("e1", "disruption", "a", ["b"], "2022-03-10", None),
    ("e2", "disruption", "a", [], "2022-03-20", None),
    ("e3", "disruption", "c", [], "2023-06-01", None),
    ("r1", "recovery", "a", [], "2022-04-01", None),
)


def test_hit_delay_and_miss():
    eps = episodes(("a", "2022-03-10", "2022-03-13", "2022-03-30"))
    sc = v.score_events(EV, eps).set_index("event_id")
    assert sc.loc["e1", "hit"] and sc.loc["e1", "hit_type"] == "new"
    assert sc.loc["e1", "delay_days"] == 3
    assert not sc.loc["e3", "hit"] and sc.loc["e3", "hit_type"] == "miss"
    assert "r1" not in sc.index  # recoveries are not targets


def test_covered_hit_for_overlapping_event():
    eps = episodes(("a", "2022-03-01", "2022-03-04", "2022-04-30"))
    sc = v.score_events(EV.iloc[[1]], eps).iloc[0]
    assert sc["hit_type"] == "covered" and sc["delay_days"] == 0


def test_early_alert_has_negative_delay():
    eps = episodes(("a", "2022-03-08", "2022-03-08", "2022-03-30"))
    assert v.score_events(EV.iloc[[0]], eps).iloc[0]["delay_days"] == -2


def test_late_alert_outside_window_is_miss():
    eps = episodes(("a", "2022-04-20", "2022-04-20", "2022-05-10"))
    assert not v.score_events(EV.iloc[[0]], eps).iloc[0]["hit"]


def test_false_alarm_labelling_uses_related_chokepoints():
    eps = episodes(
        ("a", "2022-03-10", "2022-03-12", "2022-03-20"),
        ("b", "2022-03-12", "2022-03-14", "2022-03-20"),  # related: excused
        ("c", "2022-03-12", "2022-03-14", "2022-03-20"),  # unrelated: false alarm
        ("a", "2022-09-01", "2022-09-04", "2022-09-10"),  # no event nearby
    )
    out = v.label_episodes(EV, eps)
    assert list(out["false_alarm"]) == [False, False, True, True]


def test_metrics_values():
    eps = episodes(
        ("a", "2022-03-10", "2022-03-12", "2022-03-30"),
        ("a", "2022-09-01", "2022-09-04", "2022-09-10"),
    )
    m = v.metrics(EV, eps, start="2022-01-01", end="2022-12-31", years=1.0)
    assert m["n_events"] == 2 and m["hits"] == 2  # e2 covered by first episode
    assert m["recall"] == 1.0
    assert m["precision"] == 0.5 and m["false_alarms"] == 1
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["median_delay_days"] == 2
    assert m["fa_per_year"] == 1.0


def test_metrics_empty_period():
    m = v.metrics(EV, episodes(), start="2030-01-01", end="2030-12-31")
    assert m["n_events"] == 0 and np.isnan(m["recall"]) and np.isnan(m["f1"])


def test_default_grid_has_18_unique_points():
    g = v.default_grid()
    assert len(g) == 18
    assert len({tuple(sorted(p.as_dict().items())) for p in g}) == 18


def _volume():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-01", "2023-12-31")
    parts = []
    for name in ("a", "b"):
        s = pd.Series(100 + rng.normal(0, 3, len(idx)), idx)
        if name == "a":
            s["2022-06-01":"2022-06-30"] = 30.0
            s["2023-06-01":"2023-06-30"] = 30.0
        parts.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "chokepoint": name,
                    "vessel_type": "total",
                    "vessels": s.values,
                    "capacity_tons": np.nan,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_run_grid_and_select():
    ev = events(
        ("x1", "disruption", "a", ["b"], "2022-06-01", None),
        ("x2", "disruption", "a", ["b"], "2023-06-01", None),
    )
    periods = {
        "select": ("2022-01-01", "2022-12-31"),
        "report": ("2023-01-01", "2023-12-31"),
        "full": (None, None),
    }
    res = v.run_grid(_volume(), ev, periods=periods)
    assert len(res) == 18 * 3
    best = v.select_params(res, min_select_events=1)
    assert best["period"] == "select" and best["recall"] == 1.0
    rep = res[
        (res["period"] == "report")
        & (res["tau"] == best["tau"])
        & (res["k"] == best["k"])
        & (res["n"] == best["n"])
        & (res["smooth_days"] == best["smooth_days"])
    ]
    assert rep["recall"].iloc[0] == 1.0


def test_select_warns_on_few_events_and_bad_period():
    ev = events(("x1", "disruption", "a", [], "2022-06-01", None))
    res = v.run_grid(_volume(), ev, periods={"select": (None, None)})
    with pytest.warns(UserWarning, match="fragile"):
        v.select_params(res)
    with pytest.raises(ValueError):
        v.select_params(res, period="nope")


def test_run_grid_without_data_raises():
    vol = _volume().iloc[0:0]
    with pytest.raises(ValueError, match="no volume"):
        v.run_grid(vol, EV)


def test_cusum_detects_sustained_drop():
    vol = _volume()
    s = vol[vol["chokepoint"] == "a"].set_index("date")["vessels"]
    eps = v.cusum_episodes(d.compute_scores(s))
    assert len(eps) >= 1
    assert eps["start"].min() >= ts("2022-06-01")


def test_event_window_extends_to_end_date():
    ev = events(("z", "disruption", "a", [], "2022-03-10", "2022-12-31"))
    eps = episodes(("a", "2022-10-01", "2022-10-04", "2022-10-10"))
    assert not v.label_episodes(ev, eps).iloc[0]["false_alarm"]
