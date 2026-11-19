import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

from openbb_chokepoints.data import events as ev
from openbb_chokepoints.data.proxies import load_proxies


def _event(**kw):
    base = {
        "id": "e1",
        "type": "disruption",
        "name": "Test",
        "category": "conflict",
        "chokepoint": "suez_canal",
        "news_date": "2024-01-01",
        "confidence": "medium",
        "sources": [
            {"name": "A", "type": "primary", "url": "https://a"},
            {"name": "B", "type": "secondary", "url": "https://b"},
        ],
    }
    base.update(kw)
    return base


def _write(tmp_path: Path, *events) -> Path:
    p = tmp_path / "events.yaml"
    p.write_text(yaml.safe_dump({"events": list(events)}))
    return p


def test_packaged_catalog_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ev.load_events()
    assert list(df.columns) == ev.EVENT_COLUMNS
    assert df["news_date"].is_monotonic_increasing
    assert df["id"].is_unique
    assert {"disruption", "recovery"} == set(df["type"])
    assert (df["n_sources"] >= 2).all()


def test_packaged_high_confidence_have_primary_source():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ev.load_events(min_confidence="high")
    assert (df["n_primary"] >= 1).all()


def test_packaged_overlaps_are_symmetric():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            ev.load_events()
        except UserWarning as exc:  # only the known missing-primary warning allowed
            assert "no primary source" in str(exc)


def test_filters(tmp_path):
    p = _write(
        tmp_path,
        _event(id="a"),
        _event(id="b", type="recovery", confidence="high"),
        _event(id="c", chokepoint="panama_canal", related_chokepoints=["suez_canal"]),
    )
    assert list(ev.load_events(p, event_type="recovery")["id"]) == ["b"]
    assert list(ev.load_events(p, min_confidence="high")["id"]) == ["b"]
    assert set(ev.load_events(p, chokepoint="suez_canal")["id"]) == {"a", "b", "c"}
    assert list(ev.load_events(p, chokepoint="panama_canal")["id"]) == ["c"]


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"chokepoint": "narnia"}, "unknown chokepoint"),
        ({"type": "party"}, "type must be"),
        ({"confidence": "sure"}, "confidence must be"),
        ({"news_date": "not a date"}, "news_date"),
        ({"sources": [{"name": "A", "type": "primary", "url": "u"}]}, "2 sources"),
        (
            {
                "sources": [
                    {"name": "A", "type": "primary", "url": "u"},
                    {"name": "B", "type": "primary", "url": "u"},
                ]
            },
            "duplicate source",
        ),
        ({"start_date": "2024-02-01", "end_date": "2024-01-01"}, "after end_date"),
        ({"overlaps_with": ["ghost"]}, "unknown id"),
        ({"name": ""}, "missing required"),
    ],
)
def test_invalid_events_rejected(tmp_path, bad, match):
    with pytest.raises(ev.CatalogError, match=match):
        ev.load_events(_write(tmp_path, _event(**bad)))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ev.CatalogError, match="duplicate event ids"):
        ev.load_events(_write(tmp_path, _event(), _event()))


def test_high_confidence_needs_primary(tmp_path):
    sources = [
        {"name": "A", "type": "secondary", "url": "https://a"},
        {"name": "B", "type": "secondary", "url": "https://b"},
    ]
    with pytest.raises(ev.CatalogError, match="primary"):
        ev.load_events(_write(tmp_path, _event(confidence="high", sources=sources)))
    with pytest.warns(UserWarning, match="no primary"):
        ev.load_events(_write(tmp_path, _event(sources=sources)))


def test_asymmetric_overlap_warns(tmp_path):
    p = _write(tmp_path, _event(id="a", overlaps_with=["b"]), _event(id="b"))
    with pytest.warns(UserWarning, match="vice versa"):
        ev.load_events(p)


def test_empty_catalog_and_bad_args(tmp_path):
    with pytest.raises(ev.CatalogError):
        ev.load_events(_write(tmp_path))
    with pytest.raises(ValueError):
        ev.load_events(event_type="x")
    with pytest.raises(ValueError):
        ev.load_events(min_confidence="x")


def test_sources_table():
    s = ev.load_sources()
    assert list(s.columns) == ["event_id", "name", "type", "url"]
    assert s["url"].str.startswith("http").all()
    assert isinstance(s["event_id"].iloc[0], str)


def test_dates_are_timestamps():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ev.load_events()
    assert isinstance(df["news_date"].iloc[0], pd.Timestamp)


def test_proxies_load_and_consistent():
    px = load_proxies()
    assert set(px["industry_to_sector"].values()) <= set(px["sector_proxies"])
    assert all(k == k.lower().strip() for k in px["industry_to_sector"])
    assert set(px["chokepoint_proxies"]) <= set(ev.KNOWN_CHOKEPOINTS)


def test_proxies_unknown_sector_rejected(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"industry_to_sector": {"x": "ghost"}}))
    with pytest.raises(ValueError, match="ghost"):
        load_proxies(p)
