"""PortWatch data access and cache (Work package 1, Anjaneya).

Every other module in openbb-chokepoints reads traffic data through this file.
No other module calls OpenBB for traffic data directly.

Public API
----------
get_chokepoint_volume(chokepoints, start, end) -> long-format DataFrame
get_chokepoint_info()                          -> one row per chokepoint

Volume table columns (exact names):
    date, chokepoint, vessel_type, vessels, capacity_tons
    vessel_type is one of: total, cargo, tanker, container, dry_bulk,
    general_cargo, roro

Metadata on every returned table (``df.attrs``):
    data_pulled_at  ISO-8601 UTC timestamp of the pull the data came from
    data_through    ISO date (YYYY-MM-DD) of the latest day with data
    missing_days    (volume only) {chokepoint: number of calendar days with no data}

Rules:
    * Missing days stay NaN and are counted. They are never filled with zero,
      because a zero looks exactly like a closure.
    * Cache lives in ~/.openbb_chokepoints/cache/ as Parquet, keyed by dataset,
      chokepoint and pull time. A pull is reused for 24 hours. After that the
      last 60 days (the part PortWatch revises) are re-pulled and older days
      are kept.
    * If a refresh fails and a cache exists, the last good pull is served.

Chokepoint names are the 24 lower-case names in
``openbb_imf.utils.constants.ChokepointsNames`` (e.g. ``suez_canal``,
``bab_el_mandeb_strait``). Note: openbb-imf 3.0.0's own query validator only
accepts ``bab_el-mandeb_strait`` (with a hyphen), so this module always calls
OpenBB with the ``chokepointN`` IDs instead of names.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_DIR",
    "CACHE_TTL",
    "REVISION_DAYS",
    "VESSEL_TYPES",
    "VOLUME_COLUMNS",
    "get_chokepoint_volume",
    "get_chokepoint_info",
    "list_chokepoints",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CACHE_DIR = Path.home() / ".openbb_chokepoints" / "cache"
CACHE_TTL = timedelta(hours=24)
REVISION_DAYS = 60
HISTORY_START = "2019-01-01"  # PortWatch chokepoint data starts in 2019

VESSEL_TYPES: tuple[str, ...] = (
    "total",
    "cargo",
    "tanker",
    "container",
    "dry_bulk",
    "general_cargo",
    "roro",
)
VOLUME_COLUMNS: list[str] = [
    "date",
    "chokepoint",
    "vessel_type",
    "vessels",
    "capacity_tons",
]
INFO_REQUIRED_COLUMNS: list[str] = [
    "chokepoint_code",
    "name",
    "latitude",
    "longitude",
    "vessel_count_total",
    "vessel_count_tanker",
    "vessel_count_container",
    "vessel_count_general_cargo",
    "vessel_count_dry_bulk",
    "vessel_count_roro",
    "industry_top1",
    "industry_top2",
    "industry_top3",
]

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


# --------------------------------------------------------------------------- #
# Small helpers (kept as functions so tests can monkeypatch them)
# --------------------------------------------------------------------------- #


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slug(name: str) -> str:
    """'Bab el-Mandeb Strait' -> 'bab_el_mandeb_strait'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _chokepoint_ids() -> dict[str, str]:
    """Map canonical chokepoint name -> OpenBB id, e.g. suez_canal -> chokepoint1."""
    try:
        from openbb_imf.utils.constants import CHOKEPOINTS_ID_MAP
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "openbb-imf is required to fetch PortWatch data. "
            "Install it with: pip install 'openbb-imf>=3,<4'"
        ) from exc
    return {_slug(display): cid for cid, display in CHOKEPOINTS_ID_MAP.items()}


def _fetch_volume_raw(chokepoint_id: str, start: str, end: str) -> pd.DataFrame:
    """One OpenBB call for one chokepoint. Wide format, as OpenBB returns it."""
    from openbb import obb  # imported lazily so unit tests need no OpenBB

    return obb.economy.shipping.chokepoint_volume(
        chokepoint=chokepoint_id,
        start_date=start,
        end_date=end,
        provider="imf",
    ).to_df()


def _fetch_info_raw() -> pd.DataFrame:
    from openbb import obb

    return obb.economy.shipping.chokepoint_info(provider="imf").to_df()


def _iso(d: date | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _stamp_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Chokepoint names
# --------------------------------------------------------------------------- #


def list_chokepoints() -> list[str]:
    """The 24 canonical chokepoint names, in PortWatch id order."""
    return list(_chokepoint_ids())


def _resolve_chokepoints(chokepoints: str | Iterable[str] | None) -> list[str]:
    ids = _chokepoint_ids()
    if chokepoints is None or chokepoints == "all":
        return list(ids)
    names = [chokepoints] if isinstance(chokepoints, str) else list(chokepoints)
    bad = [n for n in names if n not in ids]
    if bad:
        raise ValueError(
            f"Unknown chokepoint name(s): {bad}. Expected names from "
            f"openbb_imf.utils.constants.ChokepointsNames, e.g. {list(ids)[:4]}."
        )
    seen: list[str] = []
    for n in names:  # de-duplicate, keep order
        if n not in seen:
            seen.append(n)
    return seen


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #


def _normalize_date(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.dt.normalize().astype("datetime64[ns]")


def _to_long(raw: pd.DataFrame, chokepoint: str) -> pd.DataFrame:
    """Wide OpenBB table -> long table with VOLUME_COLUMNS (no reindexing yet)."""
    raw = raw.copy()
    if "date" not in raw.columns:
        raw = raw.reset_index()
    if "date" not in raw.columns:
        raise ValueError("PortWatch response has no 'date' column or index.")

    missing = [f"vessels_{vt}" for vt in VESSEL_TYPES if f"vessels_{vt}" not in raw]
    if missing:
        raise ValueError(
            f"PortWatch response is missing expected columns: {missing}. "
            "openbb-imf may have changed its field names; check the pinned version."
        )
    missing_cap = [f"capacity_{vt}" for vt in VESSEL_TYPES if f"capacity_{vt}" not in raw]
    if missing_cap:
        logger.warning("PortWatch response lacks %s; capacity_tons will be NaN.", missing_cap)

    dates = _normalize_date(raw["date"])
    parts = []
    for vt in VESSEL_TYPES:
        cap_col = f"capacity_{vt}"
        parts.append(
            pd.DataFrame(
                {
                    "date": dates.values,
                    "chokepoint": chokepoint,
                    "vessel_type": vt,
                    "vessels": pd.to_numeric(raw[f"vessels_{vt}"], errors="coerce")
                    .astype("float64")
                    .values,
                    "capacity_tons": (
                        pd.to_numeric(raw[cap_col], errors="coerce").astype("float64").values
                        if cap_col in raw
                        else float("nan")
                    ),
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)
    long = long.drop_duplicates(subset=["date", "chokepoint", "vessel_type"], keep="last")
    return long[VOLUME_COLUMNS].sort_values(["vessel_type", "date"]).reset_index(drop=True)


def _reindex_daily(long: pd.DataFrame) -> pd.DataFrame:
    """Give every chokepoint a row for every calendar day between its first and
    last date, for every vessel type. Gaps become NaN, never zero."""
    frames = []
    for name, grp in long.groupby("chokepoint", sort=False):
        days = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        full = pd.MultiIndex.from_product([days, VESSEL_TYPES], names=["date", "vessel_type"])
        g = (
            grp.drop(columns="chokepoint")
            .set_index(["date", "vessel_type"])
            .reindex(full)
            .reset_index()
        )
        g["chokepoint"] = name
        frames.append(g[VOLUME_COLUMNS])
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _EMPTY_DTYPES.items()})
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out.sort_values(["chokepoint", "vessel_type", "date"]).reset_index(drop=True)


_EMPTY_DTYPES = {
    "date": "datetime64[ns]",
    "chokepoint": "object",
    "vessel_type": "object",
    "vessels": "float64",
    "capacity_tons": "float64",
}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _cache_dir(cache_dir: str | Path | None) -> Path:
    path = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_files(dataset: str, key: str, cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob(f"{dataset}__{key}__*.parquet"))


def _read_cache(dataset: str, key: str, cache_dir: Path) -> tuple[pd.DataFrame | None, datetime | None]:
    files = _cache_files(dataset, key, cache_dir)
    for path in reversed(files):  # newest first
        try:
            stamp = path.stem.rsplit("__", 1)[1]
            pulled_at = datetime.strptime(stamp, _STAMP_FORMAT).replace(tzinfo=timezone.utc)
            return pd.read_parquet(path), pulled_at
        except Exception as exc:  # corrupt file: try the next one
            logger.warning("Ignoring unreadable cache file %s (%s)", path, exc)
    return None, None


def _write_cache(df: pd.DataFrame, dataset: str, key: str, pulled_at: datetime, cache_dir: Path) -> None:
    final = cache_dir / f"{dataset}__{key}__{pulled_at.strftime(_STAMP_FORMAT)}.parquet"
    tmp = final.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    for old in _cache_files(dataset, key, cache_dir):
        if old != final:
            old.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #


def _load_one_chokepoint(
    name: str, chokepoint_id: str, cache_dir: Path, refresh: bool
) -> tuple[pd.DataFrame, datetime]:
    """Long (not yet reindexed) data for one chokepoint plus the time it was pulled."""
    now = _utcnow()
    cached, pulled_at = _read_cache("volume", name, cache_dir)
    if cached is not None and not refresh and now - pulled_at < CACHE_TTL:
        return cached, pulled_at

    today = _iso(now)
    try:
        if cached is None or cached.empty:
            fresh = _to_long(_fetch_volume_raw(chokepoint_id, HISTORY_START, today), name)
            if fresh.empty:
                raise ValueError(f"PortWatch returned no data for {name}.")
            merged = fresh
        else:
            # Re-pull the last 60 days of what we hold (and any gap after it);
            # keep everything older than that from the cache.
            pull_start = cached["date"].max() - timedelta(days=REVISION_DAYS)
            fresh = _to_long(_fetch_volume_raw(chokepoint_id, _iso(pull_start), today), name)
            if fresh.empty:
                raise ValueError(f"PortWatch returned no new data for {name}.")
            merged = pd.concat([cached[cached["date"] < pull_start], fresh], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["date", "chokepoint", "vessel_type"], keep="last"
            )
            merged = merged.sort_values(["vessel_type", "date"]).reset_index(drop=True)
    except Exception as exc:
        if cached is not None:
            logger.warning(
                "Refresh failed for %s (%s); serving the cache from %s.",
                name, exc, _stamp_iso(pulled_at),
            )
            return cached, pulled_at
        raise

    _write_cache(merged, "volume", name, now, cache_dir)
    return merged, now


def get_chokepoint_volume(
    chokepoints: str | Iterable[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily PortWatch traffic in long format.

    Parameters
    ----------
    chokepoints : name or list of names (e.g. "suez_canal"); None or "all" = all 24.
    start, end : inclusive date bounds; None = no bound.
    cache_dir : override the cache folder (tests use a temp folder).
    refresh : ignore the 24-hour reuse window and re-pull the last 60 days.

    Returns
    -------
    DataFrame with columns date, chokepoint, vessel_type, vessels, capacity_tons.
    Every chokepoint has a row for every calendar day between its first and last
    date, for every vessel type; days with no data are NaN. ``df.attrs`` holds
    data_pulled_at, data_through and missing_days.
    """
    names = _resolve_chokepoints(chokepoints)
    ids = _chokepoint_ids()
    folder = _cache_dir(cache_dir)

    longs, pulls = [], []
    for name in names:
        long, pulled_at = _load_one_chokepoint(name, ids[name], folder, refresh)
        longs.append(long)
        pulls.append(pulled_at)

    out = _reindex_daily(pd.concat(longs, ignore_index=True))
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    out = out.reset_index(drop=True)

    has_data = out.loc[out["vessels"].notna(), "date"]
    totals = out[out["vessel_type"] == "total"]
    missing = totals[totals["vessels"].isna()].groupby("chokepoint").size()

    out.attrs["data_pulled_at"] = _stamp_iso(min(pulls))  # oldest pull = most cautious
    out.attrs["data_through"] = _iso(has_data.max()) if len(has_data) else None
    out.attrs["missing_days"] = {n: int(missing.get(n, 0)) for n in names}
    return out


# --------------------------------------------------------------------------- #
# Info
# --------------------------------------------------------------------------- #


def get_chokepoint_info(
    *, cache_dir: str | Path | None = None, refresh: bool = False
) -> pd.DataFrame:
    """One row per chokepoint: location, yearly average vessel counts by type
    (since 2019) and the top three industries (ranked names, no shares).

    Columns: chokepoint (canonical name, matches the volume table), then
    chokepoint_code, name, latitude, longitude, vessel_count_total,
    vessel_count_tanker, vessel_count_container, vessel_count_general_cargo,
    vessel_count_dry_bulk, vessel_count_roro, industry_top1, industry_top2,
    industry_top3. ``df.attrs`` holds data_pulled_at and data_through.
    """
    folder = _cache_dir(cache_dir)
    now = _utcnow()
    cached, pulled_at = _read_cache("info", "all", folder)

    if cached is not None and not refresh and now - pulled_at < CACHE_TTL:
        df, stamp = cached, pulled_at
    else:
        try:
            raw = _fetch_info_raw()
            if "name" not in raw.columns:
                raw = raw.reset_index()
            missing = [c for c in INFO_REQUIRED_COLUMNS if c not in raw.columns]
            if missing:
                raise ValueError(
                    f"PortWatch info response is missing expected columns: {missing}. "
                    "openbb-imf may have changed its field names; check the pinned version."
                )
            df = raw[INFO_REQUIRED_COLUMNS].copy()
            df.insert(0, "chokepoint", df["name"].map(_slug))
            df = df.sort_values("chokepoint").reset_index(drop=True)
            _write_cache(df, "info", "all", now, folder)
            stamp = now
        except Exception as exc:
            if cached is None:
                raise
            logger.warning(
                "Info refresh failed (%s); serving the cache from %s.", exc, _stamp_iso(pulled_at)
            )
            df, stamp = cached, pulled_at

    df = df.copy()
    df.attrs["data_pulled_at"] = _stamp_iso(stamp)
    df.attrs["data_through"] = _iso(stamp)  # yearly averages; no daily date exists
    return df
