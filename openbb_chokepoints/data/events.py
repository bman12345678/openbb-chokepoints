"""Event catalog loader and validator (WP2).

The catalog (``events.yaml``) is hand-curated. ``load_events()`` validates it
and returns one row per event, ready for the event study (``news_date`` is
day 0) and for detection validation.

Rules enforced on load (errors unless noted):
  * unique ids, known chokepoints, known type/category/confidence;
  * ``news_date`` is a real date and ``start_date <= end_date``;
  * at least two sources; at least one primary source (an error for
    high-confidence events, a warning otherwise);
  * ``overlaps_with`` references existing ids and is symmetric (warning).
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EVENTS_PATH = Path(__file__).with_name("events.yaml")

KNOWN_CHOKEPOINTS: tuple[str, ...] = (
    "suez_canal",
    "panama_canal",
    "bosporus_strait",
    "bab_el_mandeb_strait",
    "malacca_strait",
    "strait_of_hormuz",
    "cape_of_good_hope",
    "gibraltar_strait",
    "dover_strait",
    "oresund_strait",
    "taiwan_strait",
    "korea_strait",
    "tsugaru_strait",
    "luzon_strait",
    "lombok_strait",
    "ombai_strait",
    "bohai_strait",
    "torres_strait",
    "sunda_strait",
    "makassar_strait",
    "magellan_strait",
    "yucatan_channel",
    "windward_passage",
    "mona_passage",
)
TYPES = ("disruption", "recovery")
CATEGORIES = ("blockage", "conflict", "climate", "policy", "ceasefire", "other")
CONFIDENCE = ("high", "medium", "low")
SOURCE_TYPES = ("primary", "secondary")
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}

EVENT_COLUMNS = [
    "id",
    "type",
    "name",
    "category",
    "chokepoint",
    "related_chokepoints",
    "news_date",
    "start_date",
    "end_date",
    "confidence",
    "n_sources",
    "n_primary",
    "overlaps_with",
    "verify",
    "notes",
]


class CatalogError(ValueError):
    """The event catalog violates a rule."""


def _as_date(value: object, field: str, event_id: str) -> pd.Timestamp:
    if isinstance(value, (date, pd.Timestamp)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, str):
        try:
            return pd.Timestamp(value).normalize()
        except ValueError:
            pass
    raise CatalogError(f"{event_id}: {field} must be a date, got {value!r}")


def _check_one(raw: dict, idx: int) -> dict:
    eid = str(raw.get("id") or f"<event #{idx}>")
    for field in ("id", "type", "name", "chokepoint", "news_date", "confidence"):
        if raw.get(field) in (None, ""):
            raise CatalogError(f"{eid}: missing required field '{field}'")
    if raw["type"] not in TYPES:
        raise CatalogError(f"{eid}: type must be one of {TYPES}")
    category = raw.get("category", "other")
    if category not in CATEGORIES:
        raise CatalogError(f"{eid}: category must be one of {CATEGORIES}")
    if raw["confidence"] not in CONFIDENCE:
        raise CatalogError(f"{eid}: confidence must be one of {CONFIDENCE}")
    related = list(raw.get("related_chokepoints") or [])
    for cp in [raw["chokepoint"], *related]:
        if cp not in KNOWN_CHOKEPOINTS:
            raise CatalogError(f"{eid}: unknown chokepoint '{cp}'")

    news = _as_date(raw["news_date"], "news_date", eid)
    start = raw.get("start_date")
    end = raw.get("end_date")
    start_ts = _as_date(start, "start_date", eid) if start else pd.NaT
    end_ts = _as_date(end, "end_date", eid) if end else pd.NaT
    if pd.notna(start_ts) and pd.notna(end_ts) and start_ts > end_ts:
        raise CatalogError(f"{eid}: start_date is after end_date")

    sources = raw.get("sources") or []
    if len(sources) < 2:
        raise CatalogError(f"{eid}: needs at least 2 sources, has {len(sources)}")
    for s in sources:
        if not s.get("name") or not s.get("url"):
            raise CatalogError(f"{eid}: every source needs a name and a url")
        if s.get("type") not in SOURCE_TYPES:
            raise CatalogError(f"{eid}: source type must be one of {SOURCE_TYPES}")
    urls = [s["url"] for s in sources]
    if len(set(urls)) != len(urls):
        raise CatalogError(f"{eid}: duplicate source urls")
    n_primary = sum(s["type"] == "primary" for s in sources)
    if n_primary == 0:
        msg = f"{eid}: no primary source"
        if raw["confidence"] == "high":
            raise CatalogError(msg + " (required for high confidence)")
        warnings.warn(msg, stacklevel=3)

    return {
        "id": raw["id"],
        "type": raw["type"],
        "name": raw["name"],
        "category": category,
        "chokepoint": raw["chokepoint"],
        "related_chokepoints": related,
        "news_date": news,
        "start_date": start_ts,
        "end_date": end_ts,
        "confidence": raw["confidence"],
        "n_sources": len(sources),
        "n_primary": n_primary,
        "overlaps_with": list(raw.get("overlaps_with") or []),
        "verify": bool(raw.get("verify", False)),
        "notes": " ".join(str(raw.get("notes", "")).split()),
    }


def _check_overlaps(rows: list[dict]) -> None:
    ids = {r["id"] for r in rows}
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        for other in r["overlaps_with"]:
            if other not in ids:
                raise CatalogError(f"{r['id']}: overlaps_with unknown id '{other}'")
            if other == r["id"]:
                raise CatalogError(f"{r['id']}: overlaps with itself")
            if r["id"] not in by_id[other]["overlaps_with"]:
                warnings.warn(
                    f"{r['id']} lists {other} as overlapping, but not vice versa",
                    stacklevel=3,
                )


def load_events(
    path: str | Path | None = None,
    *,
    event_type: str | None = None,
    chokepoint: str | None = None,
    min_confidence: str = "low",
) -> pd.DataFrame:
    """Load and validate the event catalog.

    Parameters
    ----------
    path
        Alternative YAML file (default: the packaged ``events.yaml``).
    event_type
        ``"disruption"`` or ``"recovery"`` to keep only that type.
    chokepoint
        Keep events whose primary or related chokepoints include this one.
    min_confidence
        ``"low"`` (all), ``"medium"`` or ``"high"``.

    Returns
    -------
    DataFrame sorted by ``news_date`` with columns ``EVENT_COLUMNS``. A
    separate ``sources`` frame is available from ``load_sources()``.
    """
    if event_type is not None and event_type not in TYPES:
        raise ValueError(f"event_type must be one of {TYPES}")
    if min_confidence not in _CONF_RANK:
        raise ValueError(f"min_confidence must be one of {CONFIDENCE}")

    doc = yaml.safe_load(Path(path or EVENTS_PATH).read_text(encoding="utf-8"))
    raw_events = (doc or {}).get("events") or []
    if not raw_events:
        raise CatalogError("catalog contains no events")
    rows = [_check_one(r, i) for i, r in enumerate(raw_events)]
    ids = [r["id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise CatalogError(f"duplicate event ids: {dupes}")
    _check_overlaps(rows)

    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df = df[df["confidence"].map(_CONF_RANK) >= _CONF_RANK[min_confidence]]
    if event_type is not None:
        df = df[df["type"] == event_type]
    if chokepoint is not None:
        keep = df.apply(
            lambda r: (
                chokepoint == r["chokepoint"] or chokepoint in r["related_chokepoints"]
            ),
            axis=1,
        )
        df = df[keep.astype(bool)] if len(df) else df
    return df.sort_values("news_date").reset_index(drop=True)


def load_sources(path: str | Path | None = None) -> pd.DataFrame:
    """One row per (event, source): ``event_id, name, type, url``."""
    doc = yaml.safe_load(Path(path or EVENTS_PATH).read_text(encoding="utf-8"))
    out = [
        {"event_id": e["id"], "name": s["name"], "type": s["type"], "url": s["url"]}
        for e in (doc or {}).get("events", [])
        for s in e.get("sources", [])
    ]
    return pd.DataFrame(out, columns=["event_id", "name", "type", "url"])
