"""Unit tests for openbb_chokepoints.data.portwatch.

These run with no OpenBB install and no internet: a fake PortWatch source
replaces the two OpenBB calls. Sahya's recorded-response integration test
(tests/integration) covers the real OpenBB call path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from openbb_chokepoints.data import portwatch as pw

IDS = {"suez_canal": "chokepoint1", "strait_of_hormuz": "chokepoint6"}
START = "2026-08-01"


def make_raw(dates, version=0) -> pd.DataFrame:
    """Wide table shaped like obb.economy.shipping.chokepoint_volume(...).to_df()."""
    n = len(dates)
    tanker = np.full(n, 10 + 100 * version)
    container = np.full(n, 20)
    general = np.full(n, 5)
    dry = np.full(n, 8)
    roro = np.full(n, 2)
    cargo = container + general + dry + roro
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "chokepoint": "Suez Canal",
            "vessels_total": cargo + tanker,
            "vessels_cargo": cargo,
            "vessels_tanker": tanker,
            "vessels_container": container,
            "vessels_general_cargo": general,
            "vessels_dry_bulk": dry,
            "vessels_roro": roro,
        }
    )
    for vt in pw.VESSEL_TYPES:
        raw[f"capacity_{vt}"] = raw[f"vessels_{vt}"] * 1000.0
    return raw


class FakeSource:
    """Stands in for PortWatch: data runs up to (clock date - 7 days)."""

    lag_days = 7

    def __init__(self, clock):
        self.clock = clock
        self.calls: list[tuple[str, str, str]] = []
        self.version = 0
        self.fail = False
        self.drop: set[date] = set()
        self.drop_column: str | None = None
        self.start_floor = pd.Timestamp(START)

    def volume(self, chokepoint_id, start, end):
        self.calls.append((chokepoint_id, start, end))
        if self.fail:
            raise RuntimeError("PortWatch is down")
        last = pd.Timestamp(self.clock["now"].date()) - timedelta(days=self.lag_days)
        days = pd.date_range(
            max(pd.Timestamp(start), self.start_floor), min(pd.Timestamp(end), last)
        )
        days = [d for d in days if d.date() not in self.drop]
        raw = make_raw(days, self.version)
        if self.drop_column:
            raw = raw.drop(columns=self.drop_column)
        return raw

    def info(self):
        if self.fail:
            raise RuntimeError("PortWatch is down")
        return pd.DataFrame(
            {
                "chokepoint_code": ["chokepoint1", "chokepoint6"],
                "name": ["Suez Canal", "Strait of Hormuz"],
                "latitude": [30.5, 26.5],
                "longitude": [32.3, 56.5],
                "vessel_count_total": [70, 85],
                "vessel_count_tanker": [10, 40],
                "vessel_count_container": [20, 15],
                "vessel_count_general_cargo": [5, 10],
                "vessel_count_dry_bulk": [8, 15],
                "vessel_count_roro": [2, 5],
                "industry_top1": ["Machinery", "Mineral fuels"],
                "industry_top2": ["Chemicals", "Chemicals"],
                "industry_top3": ["Vehicles", "Plastics"],
            }
        )


@pytest.fixture
def clock():
    return {"now": datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)}


@pytest.fixture
def source(clock, monkeypatch):
    src = FakeSource(clock)
    monkeypatch.setattr(pw, "_utcnow", lambda: clock["now"])
    monkeypatch.setattr(pw, "_chokepoint_ids", lambda: dict(IDS))
    monkeypatch.setattr(pw, "_fetch_volume_raw", src.volume)
    monkeypatch.setattr(pw, "_fetch_info_raw", src.info)
    return src


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_volume_columns_and_types(source, tmp_path):
    df = pw.get_chokepoint_volume(
        ["suez_canal", "strait_of_hormuz"], cache_dir=tmp_path
    )
    assert list(df.columns) == [
        "date",
        "chokepoint",
        "vessel_type",
        "vessels",
        "capacity_tons",
    ]
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["vessels"].dtype == "float64"
    assert df["capacity_tons"].dtype == "float64"
    assert set(df["chokepoint"]) == {"suez_canal", "strait_of_hormuz"}
    assert set(df["vessel_type"]) == {
        "total",
        "cargo",
        "tanker",
        "container",
        "dry_bulk",
        "general_cargo",
        "roro",
    }


def test_values_land_in_the_right_vessel_type(source, tmp_path):
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    one_day = df[df["date"] == df["date"].min()].set_index("vessel_type")
    assert one_day.loc["tanker", "vessels"] == 10
    assert one_day.loc["container", "vessels"] == 20
    assert one_day.loc["total", "vessels"] == 10 + 20 + 5 + 8 + 2
    assert one_day.loc["tanker", "capacity_tons"] == 10_000.0


def test_no_duplicate_rows(source, tmp_path):
    df = pw.get_chokepoint_volume(
        ["suez_canal", "strait_of_hormuz"], cache_dir=tmp_path
    )
    assert not df.duplicated(subset=["date", "chokepoint", "vessel_type"]).any()


def test_duplicate_rows_from_source_are_removed(source, tmp_path, monkeypatch):
    def dup(chokepoint_id, start, end):
        raw = source.volume(chokepoint_id, start, end)
        return pd.concat([raw, raw.iloc[[0, 1]]], ignore_index=True)

    monkeypatch.setattr(pw, "_fetch_volume_raw", dup)
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert not df.duplicated(subset=["date", "chokepoint", "vessel_type"]).any()


def test_date_filter_is_inclusive(source, tmp_path):
    df = pw.get_chokepoint_volume(
        "suez_canal", "2026-09-01", "2026-09-10", cache_dir=tmp_path
    )
    assert df["date"].min() == pd.Timestamp("2026-09-01")
    assert df["date"].max() == pd.Timestamp("2026-09-10")
    assert len(df) == 10 * len(pw.VESSEL_TYPES)


# --------------------------------------------------------------------------- #
# Missing days: NaN, counted, never zero
# --------------------------------------------------------------------------- #


def test_removed_day_comes_back_as_nan_and_is_counted(source, tmp_path):
    source.drop = {date(2026, 9, 15)}
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    gap = df[df["date"] == pd.Timestamp("2026-09-15")]
    assert len(gap) == len(pw.VESSEL_TYPES)  # row exists for every vessel type
    assert gap["vessels"].isna().all()
    assert gap["capacity_tons"].isna().all()
    assert df.attrs["missing_days"] == {"suez_canal": 1}


def test_gaps_are_never_filled_with_zero(source, tmp_path):
    source.drop = {date(2026, 9, 15), date(2026, 9, 16)}
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert (df["vessels"] == 0).sum() == 0
    assert df.attrs["missing_days"]["suez_canal"] == 2


# --------------------------------------------------------------------------- #
# Metadata stamps
# --------------------------------------------------------------------------- #


def test_attrs_stamps(source, tmp_path, clock):
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert df.attrs["data_pulled_at"] == "2026-10-28T12:00:00Z"
    assert df.attrs["data_through"] == "2026-10-21"  # clock date minus the 7-day lag


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_second_call_within_24h_uses_cache(source, tmp_path, clock):
    pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert len(source.calls) == 1
    clock["now"] += timedelta(hours=23)
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert len(source.calls) == 1  # no new network call
    assert df.attrs["data_pulled_at"] == "2026-10-28T12:00:00Z"
    assert len(list(tmp_path.glob("volume__suez_canal__*.parquet"))) == 1


def test_first_pull_requests_full_history(source, tmp_path):
    pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    assert source.calls[0] == ("chokepoint1", "2019-01-01", "2026-10-28")


def test_after_24h_repulls_last_60_days_and_keeps_older_days(source, tmp_path, clock):
    first = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    through = pd.Timestamp(first.attrs["data_through"])

    source.version = 1  # PortWatch now reports different (revised) numbers
    clock["now"] += timedelta(hours=25)
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)

    pull_start = through - timedelta(days=60)
    assert source.calls[-1] == (
        "chokepoint1",
        pull_start.strftime("%Y-%m-%d"),
        "2026-10-29",
    )

    tanker = df[df["vessel_type"] == "tanker"].set_index("date")["vessels"]
    assert tanker[pull_start - timedelta(days=1)] == 10  # older day kept from cache
    assert tanker[pull_start] == 110  # revised window replaced
    assert tanker[through] == 110
    assert df.attrs["data_pulled_at"] == "2026-10-29T13:00:00Z"
    assert not df.duplicated(subset=["date", "chokepoint", "vessel_type"]).any()
    assert (
        len(list(tmp_path.glob("volume__suez_canal__*.parquet"))) == 1
    )  # old pull removed


def test_refresh_flag_forces_repull(source, tmp_path):
    pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path, refresh=True)
    assert len(source.calls) == 2


def test_failed_refresh_serves_last_good_cache(source, tmp_path, clock):
    first = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    source.fail = True
    clock["now"] += timedelta(days=3)
    df = pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(df, first, check_exact=True)
    assert df.attrs["data_pulled_at"] == "2026-10-28T12:00:00Z"  # honest about its age


def test_failure_with_no_cache_raises(source, tmp_path):
    source.fail = True
    with pytest.raises(RuntimeError, match="down"):
        pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)


def test_each_chokepoint_is_cached_separately(source, tmp_path):
    pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)
    pw.get_chokepoint_volume(["suez_canal", "strait_of_hormuz"], cache_dir=tmp_path)
    assert [c[0] for c in source.calls] == ["chokepoint1", "chokepoint6"]


# --------------------------------------------------------------------------- #
# Fail loudly
# --------------------------------------------------------------------------- #


def test_renamed_source_column_fails_loudly(source, tmp_path):
    source.drop_column = "vessels_tanker"
    with pytest.raises(ValueError, match="vessels_tanker"):
        pw.get_chokepoint_volume("suez_canal", cache_dir=tmp_path)


def test_unknown_chokepoint_name_raises(source, tmp_path):
    with pytest.raises(ValueError, match="Unknown chokepoint"):
        pw.get_chokepoint_volume("not_a_strait", cache_dir=tmp_path)


def test_none_means_all_chokepoints(source, tmp_path):
    df = pw.get_chokepoint_volume(None, cache_dir=tmp_path)
    assert set(df["chokepoint"]) == set(IDS)


# --------------------------------------------------------------------------- #
# Info
# --------------------------------------------------------------------------- #


def test_info_columns_and_join_key(source, tmp_path):
    info = pw.get_chokepoint_info(cache_dir=tmp_path)
    assert list(info.columns) == ["chokepoint", *pw.INFO_REQUIRED_COLUMNS]
    assert set(info["chokepoint"]) == {"suez_canal", "strait_of_hormuz"}
    assert info.attrs["data_pulled_at"] == "2026-10-28T12:00:00Z"
    assert info.attrs["data_through"] == "2026-10-28"


def test_info_is_cached_for_24h(source, tmp_path, clock, monkeypatch):
    calls = []
    real = source.info
    monkeypatch.setattr(pw, "_fetch_info_raw", lambda: calls.append(1) or real())
    pw.get_chokepoint_info(cache_dir=tmp_path)
    clock["now"] += timedelta(hours=1)
    pw.get_chokepoint_info(cache_dir=tmp_path)
    assert len(calls) == 1
    clock["now"] += timedelta(hours=24)
    pw.get_chokepoint_info(cache_dir=tmp_path)
    assert len(calls) == 2


def test_info_missing_column_fails_loudly(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        pw, "_fetch_info_raw", lambda: source.info().drop(columns="industry_top2")
    )
    with pytest.raises(ValueError, match="industry_top2"):
        pw.get_chokepoint_info(cache_dir=tmp_path)
