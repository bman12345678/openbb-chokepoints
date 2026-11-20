"""OpenBB router for the chokepoints extension (obb.chokepoints).

Commands here are thin wrappers: fetch data, call the pure functions in
``openbb_chokepoints.core``, and wrap the results in an OBBject.
"""

import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Query
from openbb_core.app.model.abstract.error import OpenBBError
from openbb_core.app.model.abstract.warning import OpenBBWarning
from openbb_core.app.model.example import APIEx, PythonEx
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.router import Router

from openbb_chokepoints.core.event_study import (
    DEFAULT_ESTIMATION_WINDOW,
    DEFAULT_EVENT_WINDOW,
    DEFAULT_MIN_ESTIMATION_OBS,
    run_event_study,
)
from openbb_chokepoints.data.market import get_returns, history_range
from openbb_chokepoints.models import (
    EventCatalogData,
    EventStudyData,
    ExposureData,
    MonitorData,
    OptimizeData,
    OptimizeParameters,
    StressTestData,
    StressTestParameters,
)

router = Router(prefix="", description="Chokepoint disruption risk analytics.")

DISCLAIMER = "Scenario analysis based on past events, not investment advice."
VesselType = Literal[
    "total",
    "cargo",
    "tanker",
    "container",
    "dry_bulk",
    "general_cargo",
    "roro",
]
EventType = Literal[
    "blockage", "conflict", "climate", "policy", "ceasefire", "other", "detected"
]


@router.command(
    methods=["GET"],
    widget_config={
        "name": "Event Study",
        "description": (
            "Measure abnormal market returns around a dated chokepoint event."
        ),
        "category": "Chokepoint Risk",
        "type": "chart",
    },
    mcp_config={"name": "event_study", "tags": ["chokepoints"]},
    examples=[
        APIEx(parameters={"symbols": "ZIM,FRO", "event_date": "2023-12-15"}),
        APIEx(
            description="Non-US listing: use its local index as the market.",
            parameters={
                "symbols": "MAERSK-B.CO",
                "event_date": "2023-12-15",
                "market_symbol": "^OMXC25",
                "min_estimation_obs": 100,
            },
        ),
        PythonEx(
            description="Container and tanker shipping around the Red Sea"
            " diversions of December 2023.",
            code=[
                "res = obb.chokepoints.event_study(",
                "    symbols=['ZIM', 'FRO'],",
                "    event_date='2023-12-15',",
                "    event_window=(-5, 20),",
                ")",
                "res.to_df()[['symbol', 'window', 'car', 'car_t', 'p_value']]",
            ],
        ),
    ],
)
def event_study(
    symbols: Annotated[
        str | list[str],
        Query(
            description="Tickers to study. A list or a comma-separated string,"
            " e.g. 'ZIM,FRO'."
        ),
    ],
    event_date: Annotated[
        date,
        Query(
            description="Event date (YYYY-MM-DD). Day 0 is each symbol's first"
            " trading day on or after it."
        ),
    ],
    market_symbol: Annotated[
        str,
        Query(
            description="Market index for the market model. Use the local index"
            " for non-US listings (e.g. '^OMXC25' for Copenhagen, '^N225' for"
            " Tokyo); the default suits US stocks."
        ),
    ] = "SPY",
    estimation_window: Annotated[
        tuple[int, int],
        Query(
            description="Inclusive (start, end) trading days relative to day 0"
            " used to fit the market model. Must end before the event window."
        ),
    ] = DEFAULT_ESTIMATION_WINDOW,
    event_window: Annotated[
        tuple[int, int],
        Query(
            description="Inclusive (start, end) trading days relative to day 0 over"
            " which abnormal returns are measured. CARs are also reported for"
            " (-1, 1), (0, 5) and (0, 20) when they fit inside it."
        ),
    ] = DEFAULT_EVENT_WINDOW,
    min_estimation_obs: Annotated[
        int,
        Query(
            ge=3,
            description="Skip a symbol (with a warning) if fewer estimation"
            " days are available.",
        ),
    ] = DEFAULT_MIN_ESTIMATION_OBS,
) -> OBBject[list[EventStudyData]]:
    """Market-model event study: abnormal returns and CAR t-tests around a date.

    Fits R_i = alpha + beta * R_m by OLS over the estimation window, then
    measures abnormal returns AR = R_i - (alpha + beta * R_m) over the event
    window. CAR t-stats include the estimation error in alpha and beta
    (MacKinlay 1997). Prices come from Yahoo Finance, adjusted for splits and
    dividends. Returns are decimals.
    """
    tickers = _split_symbols(symbols)
    market = market_symbol.strip().upper()
    estimation_window = tuple(estimation_window)
    event_window = tuple(event_window)

    start, end = history_range(event_date, estimation_window, event_window)
    returns = get_returns([*tickers, market], start, end)
    if market not in returns.columns:
        raise ValueError(f"No price data for market symbol '{market}'.")

    if market in tickers:
        warnings.warn(
            f"{market} is the market symbol; skipped.", OpenBBWarning, stacklevel=2
        )
    assets = returns[[t for t in tickers if t in returns.columns and t != market]]
    for missing in sorted(set(tickers) - set(assets.columns) - {market}):
        warnings.warn(
            f"{missing}: no price data; skipped.", OpenBBWarning, stacklevel=2
        )

    rows = run_event_study(
        returns_df=assets,
        market_series=returns[market],
        event_date=event_date,
        est_window=estimation_window,
        evt_window=event_window,
        min_obs=min_estimation_obs,
    )
    data_pulled_at = datetime.now(timezone.utc)
    data_through = returns.index.max().date() if not returns.empty else None
    warnings.warn(DISCLAIMER, OpenBBWarning, stacklevel=2)
    return OBBject(
        results=[
            EventStudyData.model_validate(
                {**row, "data_pulled_at": data_pulled_at, "data_through": data_through}
            )
            for row in rows
        ],
        provider="yfinance",
    )


@router.command(
    methods=["GET"],
    widget_config={
        "name": "Traffic Monitor",
        "description": "Rank abnormal chokepoint traffic and show its history.",
        "category": "Chokepoint Risk",
        "type": "chart",
    },
    mcp_config={"name": "monitor", "tags": ["chokepoints"]},
    examples=[
        APIEx(parameters={"chokepoint": "bab_el_mandeb_strait"}),
        PythonEx(
            description="Review container traffic through Bab el-Mandeb.",
            code=[
                "res = obb.chokepoints.monitor(",
                "    chokepoint='bab_el_mandeb_strait', vessel_type='container'",
                ")",
                "res.to_df().tail()",
            ],
        ),
    ],
)
def monitor(
    chokepoint: str | None = None,
    vessel_type: VesselType | None = None,
    lookback_days: int = 90,
    threshold: float = 3.0,
) -> OBBject[list[MonitorData]]:
    """Track abnormal ship traffic and return the plotted daily history."""
    try:
        from openbb_chokepoints.core.detection import DetectionParams, compute_scores
        from openbb_chokepoints.data.portwatch import get_chokepoint_volume
    except ImportError as exc:
        raise OpenBBError(
            "The traffic monitor needs the PortWatch adapter and detection core. "
            "Merge those team modules before calling this route."
        ) from exc

    end = date.today()
    start = end - timedelta(days=lookback_days + 430)
    chokepoints = [chokepoint] if chokepoint else None
    volume = get_chokepoint_volume(chokepoints, start, end)
    if volume.empty:
        raise OpenBBError("PortWatch returned no traffic for the requested period.")
    pulled_at = volume.attrs.get("data_pulled_at")
    data_through = volume.attrs.get("data_through")
    rows = []
    for (name, category), group in volume.groupby(
        ["chokepoint", "vessel_type"], sort=True
    ):
        if vessel_type and category != vessel_type:
            continue
        group = group.sort_values("date").set_index("date")
        series = group["vessels"].astype(float)
        scores = compute_scores(
            series,
            DetectionParams(
                smooth_days=7,
                baseline_days=lookback_days,
                min_baseline_obs=max(2, min(45, lookback_days)),
                tau=threshold,
            ),
        )
        for timestamp, score in scores.iterrows():
            center = score["baseline_median"]
            current = score["smoothed"]
            if pd.isna(current):
                continue
            rows.append(
                {
                    "chokepoint": name,
                    "vessel_type": category,
                    "as_of": pd.Timestamp(timestamp).date(),
                    "avg_7d": float(current),
                    "baseline_median": None if pd.isna(center) else float(center),
                    "normal_low": None
                    if pd.isna(center)
                    else max(0.0, float(center - threshold * score["baseline_scale"])),
                    "normal_high": None
                    if pd.isna(center)
                    else float(center + threshold * score["baseline_scale"]),
                    "robust_z": None if pd.isna(score["z"]) else float(score["z"]),
                    "pct_vs_baseline": None
                    if pd.isna(center) or center == 0
                    else float(current / center - 1),
                    "pct_vs_last_year": None
                    if pd.isna(score["yoy"])
                    else float(score["yoy"]),
                    "flagged": bool(score["flag"]),
                    "low_traffic": bool(
                        pd.notna(center) and score["baseline_scale"] == 1.0
                    ),
                    "data_pulled_at": pulled_at,
                    "data_through": data_through,
                }
            )
    rows.sort(key=lambda row: (row["chokepoint"], row["vessel_type"], row["as_of"]))
    return OBBject(
        results=[MonitorData.model_validate(row) for row in rows], provider="imf"
    )


@router.command(
    methods=["GET"],
    widget_config={
        "name": "Event Catalog",
        "description": "Browse dated disruption and recovery events.",
        "category": "Chokepoint Risk",
        "type": "chart",
    },
    mcp_config={"name": "events", "tags": ["chokepoints"]},
    examples=[
        APIEx(
            description="List Red Sea disruption and recovery events.",
            parameters={
                "chokepoint": "bab_el_mandeb_strait",
                "event_type": "conflict",
                "direction": "disruption",
                "source": "catalog",
            },
        ),
        PythonEx(
            description="Review confirmed Red Sea conflict events.",
            code=[
                "res = obb.chokepoints.events(",
                "    chokepoint='bab_el_mandeb_strait', event_type='conflict'",
                ")",
                "res.to_df()[['event_id', 'news_date', 'direction', 'confidence']]",
            ],
        ),
    ],
)
def events(
    chokepoint: str | None = None,
    event_type: EventType | None = None,
    direction: Literal["disruption", "recovery"] | None = None,
    source: Literal["catalog", "detected", "both"] = "both",
) -> OBBject[list[EventCatalogData]]:
    """List dated catalog events and optionally traffic-detected episodes."""
    try:
        from openbb_chokepoints.data.events import load_events, load_sources
    except ImportError as exc:
        raise OpenBBError(
            "The event catalog loader is not available in this checkout yet."
        ) from exc

    records = []
    if source in {"catalog", "both"}:
        # Anjaneya's catalog uses ``type`` for disruption/recovery and
        # ``category`` for blockage/conflict/climate/policy/other. The data
        # adapter currently names its type filter ``event_type``.
        catalog = load_events(
            chokepoint=chokepoint,
            event_type=direction,
            min_confidence="low",
        )
        sources = load_sources()
        sources_by_event = sources.groupby("event_id")["url"].apply(list).to_dict()
        records.extend(
            _catalog_event_row(
                record,
                sources=sources_by_event.get(record["id"], []),
            )
            for record in catalog.to_dict(orient="records")
        )
    if source in {"detected", "both"}:
        records.extend(_detected_event_rows(chokepoint))
    rows = records
    if event_type is not None:
        rows = [row for row in rows if row.get("event_type") == event_type]
    if direction is not None:
        rows = [row for row in rows if row.get("direction") == direction]
    pulled_at = datetime.now(timezone.utc)
    data_through = max(
        (pd.Timestamp(row["news_date"]).date() for row in rows), default=None
    )
    rows = [
        {**row, "data_pulled_at": pulled_at, "data_through": data_through}
        for row in rows
    ]
    return OBBject(
        results=[EventCatalogData.model_validate(row) for row in rows], provider="imf"
    )


@router.command(
    methods=["GET"],
    widget_config={
        "name": "Exposure Lookup",
        "description": "Show vessel shares, industries, and market proxy symbols.",
        "category": "Chokepoint Risk",
        "type": "chart",
    },
    mcp_config={"name": "exposure", "tags": ["chokepoints"]},
    examples=[
        APIEx(parameters={"chokepoint": "suez_canal"}),
        PythonEx(
            description="Inspect vessel mix and industry rankings at Suez.",
            code=[
                "res = obb.chokepoints.exposure(chokepoint='suez_canal')",
                "res.to_df()",
            ],
        ),
    ],
)
def exposure(
    chokepoint: str,
    measure: Literal["share_count", "share_capacity"] = "share_count",
) -> OBBject[list[ExposureData]]:
    """Show vessel mix, PortWatch industry ranks, and mapped market proxies."""
    try:
        from openbb_chokepoints.data.portwatch import (
            get_chokepoint_info,
            get_chokepoint_volume,
        )
    except ImportError as exc:
        raise OpenBBError(
            "Exposure lookup needs the PortWatch adapter. "
            "Merge that data module before calling this route."
        ) from exc

    info = get_chokepoint_info()
    match = info[
        info["name"].astype(str).map(_canonical_key).eq(_canonical_key(chokepoint))
    ]
    if match.empty and "chokepoint_code" in info:
        match = info[
            info["chokepoint_code"]
            .astype(str)
            .map(_canonical_key)
            .eq(_canonical_key(chokepoint))
        ]
    if match.empty:
        raise OpenBBError(f"No PortWatch exposure row for '{chokepoint}'.")
    item = match.iloc[0]
    total_value = item.get("vessel_count_total")
    total = float(total_value) if pd.notna(total_value) else 0.0
    end = date.today()
    recent = get_chokepoint_volume([chokepoint], end - timedelta(days=365), end)
    recent_pull = recent.attrs.get("data_pulled_at")
    recent_through = recent.attrs.get("data_through")
    proxies = _load_proxy_map()
    share_types = {"tanker", "container", "dry_bulk", "general_cargo", "roro"}
    recent_vessels = (
        recent[recent["vessel_type"].isin(share_types)]
        if not recent.empty and "vessel_type" in recent
        else pd.DataFrame()
    )
    rows = []
    for vessel_type in ("tanker", "container", "dry_bulk", "general_cargo", "roro"):
        count = item.get(f"vessel_count_{vessel_type}")
        matching = (
            recent[recent["vessel_type"].eq(vessel_type)]
            if not recent.empty and "vessel_type" in recent
            else pd.DataFrame()
        )
        count_total = (
            float(recent_vessels["vessels"].sum()) if not recent_vessels.empty else 0.0
        )
        capacity_total = (
            float(recent_vessels["capacity_tons"].sum())
            if not recent_vessels.empty and "capacity_tons" in recent_vessels
            else 0.0
        )
        count_recent = (
            float(matching["vessels"].sum()) if not recent_vessels.empty else None
        )
        capacity_recent = (
            float(matching["capacity_tons"].sum())
            if not recent_vessels.empty and "capacity_tons" in matching
            else None
        )
        rows.append(
            {
                "chokepoint": chokepoint,
                "category": "vessel_type",
                "name": vessel_type,
                "share_count": (
                    count_recent / count_total
                    if count_recent is not None and count_total > 0
                    else (None if not total or pd.isna(count) else float(count) / total)
                ),
                "share_capacity": (
                    capacity_recent / capacity_total
                    if capacity_recent is not None and capacity_total > 0
                    else None
                ),
                "rank": None,
                "proxy_symbols": proxies.get(vessel_type, []),
                "data_pulled_at": recent_pull,
                "data_through": recent_through,
            }
        )
    for rank in range(1, 4):
        industry = item.get(f"industry_top{rank}")
        if industry is not None and not pd.isna(industry):
            rows.append(
                {
                    "chokepoint": chokepoint,
                    "category": "industry",
                    "name": str(industry),
                    "share_count": None,
                    "share_capacity": None,
                    "rank": rank,
                    "proxy_symbols": proxies.get(_canonical_key(industry), []),
                    "data_pulled_at": recent_pull,
                    "data_through": recent_through,
                }
            )
    route_proxies = proxies.get(f"chokepoint:{_canonical_key(chokepoint)}", [])
    if route_proxies:
        rows.append(
            {
                "chokepoint": chokepoint,
                "category": "transport",
                "name": "Chokepoint-linked companies",
                "share_count": None,
                "share_capacity": None,
                "rank": None,
                "proxy_symbols": route_proxies,
                "data_pulled_at": recent_pull,
                "data_through": recent_through,
            }
        )
    return OBBject(
        results=[ExposureData.model_validate(row) for row in rows], provider="imf"
    )


@router.command(
    methods=["POST"],
    widget_config={
        "name": "Portfolio Stress Test",
        "description": "Apply historical chokepoint event returns to a portfolio.",
        "category": "Chokepoint Risk",
        "type": "omni",
    },
    mcp_config={"name": "stress_test", "tags": ["chokepoints"]},
    examples=[
        APIEx(
            description="Stress a balanced shipping-stock and broad-market portfolio.",
            parameters={
                "data": [
                    {"symbol": "ZIM", "weight": 0.5},
                    {"symbol": "SPY", "weight": 0.5},
                ],
                "chokepoint": "bab_el_mandeb_strait",
                "event_type": "conflict",
            },
        ),
        PythonEx(
            description="Stress test a two-holding portfolio.",
            code=[
                "rows = [{'symbol': 'ZIM', 'weight': 0.5},",
                "        {'symbol': 'SPY', 'weight': 0.5}]",
                "res = obb.chokepoints.stress_test(",
                "    data=rows, chokepoint='bab_el_mandeb_strait',",
                "    event_type='conflict'",
                ")",
                "res.to_df()",
            ],
        ),
    ],
)
def stress_test(parameters: StressTestParameters) -> OBBject[list[StressTestData]]:
    """Estimate portfolio returns across comparable past chokepoint events."""
    try:
        from openbb_chokepoints.core.stress import stress_test as calculate_stress
    except ImportError as exc:
        raise OpenBBError(
            "Portfolio stress testing needs the analytics core module."
        ) from exc

    symbols, weights, events, car_matrix, raw_matrix, data_through = (
        _collect_event_cars(
            parameters.data,
            chokepoint=parameters.chokepoint,
            event_type=parameters.event_type,
            direction=parameters.direction,
            window=parameters.window,
        )
    )
    result = calculate_stress(weights, car_matrix)
    rows = _stress_result_rows(
        result, symbols, weights, events, car_matrix, raw_matrix, data_through
    )
    warnings.warn(DISCLAIMER, OpenBBWarning, stacklevel=2)
    return OBBject(
        results=[StressTestData.model_validate(row) for row in rows],
        provider="yfinance",
    )


@router.command(
    methods=["POST"],
    widget_config={
        "name": "Exposure Optimizer",
        "description": "Find portfolio weights within a scenario loss limit.",
        "category": "Chokepoint Risk",
        "type": "omni",
    },
    mcp_config={"name": "optimize", "tags": ["chokepoints"]},
    examples=[
        APIEx(
            description="Reduce the historical conflict loss to three percent.",
            parameters={
                "data": [
                    {"symbol": "ZIM", "weight": 0.5},
                    {"symbol": "SPY", "weight": 0.5},
                ],
                "chokepoint": "bab_el_mandeb_strait",
                "event_type": "conflict",
                "max_scenario_loss": 0.03,
            },
        ),
        PythonEx(
            description="Find a portfolio change under a three-percent limit.",
            code=[
                "rows = [{'symbol': 'ZIM', 'weight': 0.5},",
                "        {'symbol': 'SPY', 'weight': 0.5}]",
                "res = obb.chokepoints.optimize(",
                "    data=rows, chokepoint='bab_el_mandeb_strait',",
                "    event_type='conflict', max_scenario_loss=0.03",
                ")",
                "res.to_df()",
            ],
        ),
    ],
)
def optimize(parameters: OptimizeParameters) -> OBBject[list[OptimizeData]]:
    """Find portfolio changes that keep historical chokepoint loss within a limit."""
    try:
        from openbb_chokepoints.core.optimize import min_change, min_variance
    except ImportError as exc:
        raise OpenBBError(
            "Portfolio optimization needs the analytics core module."
        ) from exc

    symbols, weights, events, car_matrix, _, data_through = _collect_event_cars(
        parameters.data,
        chokepoint=parameters.chokepoint,
        event_type=parameters.event_type,
        direction=parameters.direction,
        window=(0, 20),
    )
    reactions = (
        car_matrix.mean(axis=1).to_numpy()
        if parameters.constraint == "average"
        else car_matrix.T.to_numpy()
    )
    bounds = [(0.0, 1.0)] * len(symbols)
    from openbb_chokepoints.data.market import get_covariance

    covariance = get_covariance(symbols, lookback_days=252)
    data_through = covariance.attrs.get("data_through", data_through)
    if parameters.objective == "min_variance":
        new_weights = min_variance(
            covariance.to_numpy(), reactions, parameters.max_scenario_loss, bounds
        )
    else:
        new_weights = min_change(
            weights, reactions, parameters.max_scenario_loss, bounds
        )
    before = car_matrix.to_numpy().T @ weights
    after = car_matrix.to_numpy().T @ np.asarray(new_weights)
    loss_before = float(np.median(before))
    loss_after = float(np.median(after))
    loss_min_before = float(np.min(before))
    loss_max_before = float(np.max(before))
    loss_min_after = float(np.min(after))
    loss_max_after = float(np.max(after))
    turnover = float(np.abs(np.asarray(new_weights) - weights).sum() / 2)
    weight_delta = np.asarray(new_weights) - weights
    tracking_error = float(
        np.sqrt(max(weight_delta @ covariance.to_numpy() @ weight_delta, 0) * 252)
    )
    data_pulled_at = covariance.attrs.get("data_pulled_at") or datetime.now(
        timezone.utc
    )
    rows = [
        {
            "symbol": symbol,
            "current_weight": float(old),
            "new_weight": float(new),
            "change": float(new - old),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_min_before": loss_min_before,
            "loss_max_before": loss_max_before,
            "loss_min_after": loss_min_after,
            "loss_max_after": loss_max_after,
            "turnover": turnover,
            "tracking_error": tracking_error,
            "objective": parameters.objective,
            "constraint": parameters.constraint,
            "data_pulled_at": data_pulled_at,
            "data_through": data_through,
        }
        for symbol, old, new in zip(symbols, weights, new_weights, strict=True)
    ]
    warnings.warn(DISCLAIMER, OpenBBWarning, stacklevel=2)
    return OBBject(
        results=[OptimizeData.model_validate(row) for row in rows], provider="yfinance"
    )


def _split_symbols(symbols: str | list[str]) -> list[str]:
    items = [symbols] if isinstance(symbols, str) else symbols
    tickers = [s.strip().upper() for item in items for s in item.split(",")]
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        raise ValueError("Provide at least one symbol.")
    return tickers


def _as_dict(value):
    """Convert catalog records from dicts, Pydantic models, or dataclasses."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return vars(value)
    return dict(value)


def _catalog_event_row(record, *, sources=None):
    """Adapt Anjaneya's catalog schema to the public output-model schema."""
    end_date = record.get("end_date")
    if pd.isna(end_date):
        end_date = None
    return {
        "event_id": record["id"],
        "news_time_utc": record.get("news_time_utc"),
        "chokepoint": record["chokepoint"],
        "event_type": record.get("category", "other"),
        "direction": record["type"],
        "news_date": record["news_date"],
        "end_date": end_date,
        "confidence": record["confidence"],
        "description": record["name"],
        "sources": record.get("sources", []) if sources is None else sources,
        "related_chokepoints": record.get("related_chokepoints", []),
        "overlaps_with": record.get("overlaps_with", []),
    }


def _detected_event_rows(chokepoint):
    """Represent detected low-traffic episodes as low-confidence events."""
    from openbb_chokepoints.core.detection import DetectionParams, detect
    from openbb_chokepoints.data.portwatch import get_chokepoint_volume

    volume = get_chokepoint_volume(
        [chokepoint] if chokepoint else None,
        date(2019, 1, 1),
        date.today(),
    )
    if volume.empty:
        return []
    rows = []
    totals = volume[volume["vessel_type"] == "total"]
    for name, group in totals.groupby("chokepoint", sort=True):
        _, episodes = detect(group.set_index("date")["vessels"], DetectionParams())
        for episode in episodes.to_dict(orient="records"):
            onset = pd.Timestamp(episode["onset"]).date()
            end = pd.Timestamp(episode["end"]).date()
            rows.append(
                {
                    "event_id": f"detected_{name}_{onset.isoformat()}",
                    "chokepoint": name,
                    "event_type": "detected",
                    "direction": "disruption",
                    "news_date": onset,
                    "end_date": end,
                    "confidence": "low",
                    "description": "Traffic drop detected by PortWatch monitor",
                    "sources": [],
                    "related_chokepoints": [],
                    "overlaps_with": [],
                }
            )
    return rows


def _canonical_key(value):
    """Compare PortWatch slugs and display labels consistently."""
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _load_proxy_map() -> dict[str, list[str]]:
    """Read optional vessel/industry proxy mappings from the data package."""
    path = Path(__file__).parent / "data" / "proxies.yaml"
    if not path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result = {}
    # Current data branch schema: industry labels map to sectors, then to
    # public-market proxies; chokepoint proxies are route-linked companies.
    industry_to_sector = raw.get("industry_to_sector", {})
    sector_proxies = raw.get("sector_proxies", {})
    if isinstance(industry_to_sector, dict) and isinstance(sector_proxies, dict):
        for industry, sector in industry_to_sector.items():
            result[_canonical_key(industry)] = [
                str(symbol) for symbol in sector_proxies.get(sector, []) or []
            ]
    for name, symbols in (raw.get("chokepoint_proxies", {}) or {}).items():
        result[f"chokepoint:{_canonical_key(name)}"] = [
            str(symbol) for symbol in symbols or []
        ]
    for category in ("vessel_types", "industries", "vessel_type", "industry"):
        values = raw.get(category, {}) if isinstance(raw, dict) else {}
        if isinstance(values, dict):
            for name, symbols in values.items():
                if isinstance(symbols, dict):
                    symbols = symbols.get("symbols", symbols.get("proxy_symbols", []))
                result[str(name)] = [str(symbol) for symbol in symbols or []]
    if isinstance(raw, dict):
        for name, symbols in raw.items():
            if name in {"vessel_types", "industries", "vessel_type", "industry"}:
                continue
            if isinstance(symbols, dict):
                symbols = symbols.get("symbols", symbols.get("proxy_symbols", []))
            if isinstance(symbols, list):
                result.setdefault(str(name), [str(symbol) for symbol in symbols])
    return result


def _collect_event_cars(data, *, chokepoint, event_type, direction, window):
    """Build an asset-by-event CAR matrix from the catalog and price loader."""
    try:
        from openbb_chokepoints.data.events import load_events
    except ImportError as exc:
        raise OpenBBError(
            "Portfolio scenario commands need the event catalog loader."
        ) from exc

    symbols = [holding.symbol.strip().upper() for holding in data]
    if len(symbols) != len(set(symbols)):
        raise OpenBBError("Portfolio contains a repeated symbol.")
    weights = np.asarray([holding.weight for holding in data], dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise OpenBBError("Portfolio weights must be finite and have a positive total.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        weights = weights / weights.sum()
        warnings.warn(
            "Portfolio weights were normalized to sum to 1.",
            OpenBBWarning,
            stacklevel=3,
        )

    catalog = load_events(
        chokepoint=chokepoint,
        event_type=direction,
        min_confidence="medium",
    )
    events = [
        _catalog_event_row(record)
        for record in catalog.to_dict(orient="records")
        if event_type is None or record.get("category") == event_type
    ]
    if not events:
        raise OpenBBError("No comparable historical events match this scenario.")

    car_columns = {}
    raw_columns = {}
    latest = None
    skipped = []
    for event in events:
        event_date = pd.Timestamp(event.get("news_date")).date()
        event_id = str(event.get("event_id", event_date))
        start, end = history_range(event_date, DEFAULT_ESTIMATION_WINDOW, window)
        returns = get_returns([*symbols, "SPY"], start, end)
        if "SPY" not in returns:
            skipped.append(event_id)
            continue
        rows = run_event_study(
            returns_df=returns[[symbol for symbol in symbols if symbol in returns]],
            market_series=returns["SPY"],
            event_date=event_date,
            est_window=DEFAULT_ESTIMATION_WINDOW,
            evt_window=window,
            min_obs=DEFAULT_MIN_ESTIMATION_OBS,
            sub_windows=(),
        )
        full_label = f"({window[0]}, {window[1]})"
        by_symbol = {
            row["symbol"]: row["car"] for row in rows if row.get("window") == full_label
        }
        if not set(symbols) <= by_symbol.keys():
            skipped.append(event_id)
            continue
        car_columns[event_id] = [by_symbol[symbol] for symbol in symbols]
        raw_columns[event_id] = [
            _raw_event_return(returns[symbol], event_date, window) for symbol in symbols
        ]
        if not returns.empty:
            latest = (
                max(latest, returns.index.max().date())
                if latest
                else returns.index.max().date()
            )
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} historical events with incomplete market data.",
            OpenBBWarning,
            stacklevel=3,
        )
    if not car_columns:
        raise OpenBBError(
            "No matching event has complete return history for every holding."
        )
    matrix = pd.DataFrame(car_columns, index=symbols, dtype=float)
    raw_matrix = pd.DataFrame(raw_columns, index=symbols, dtype=float)
    included_events = [
        event
        for event in events
        if str(event.get("event_id", event.get("news_date"))) in matrix.columns
    ]
    return symbols, weights, included_events, matrix, raw_matrix, latest


def _stress_result_rows(
    result, symbols, weights, events, car_matrix, raw_matrix, data_through
):
    """Convert the core stress result contract to OpenBB table rows."""
    if not isinstance(result, dict):
        raise TypeError(
            "core.stress.stress_test must return a mapping with per_event, "
            "summary, and contributions entries."
        )
    per_event = result["per_event"]
    summary = result["summary"]
    contributions = result["contributions"]
    per_event = per_event.to_dict() if hasattr(per_event, "to_dict") else per_event
    if isinstance(contributions, pd.DataFrame):
        contributions = contributions.to_dict(orient="index")
    elif hasattr(contributions, "to_dict"):
        contributions = contributions.to_dict()
    data_pulled_at = datetime.now(timezone.utc)
    rows = []
    for event in events:
        event_id = str(event.get("event_id", event.get("news_date")))
        if event_id not in per_event:
            continue
        portfolio_car = float(per_event[event_id])
        raw_values = raw_matrix[event_id].to_numpy(dtype=float)
        portfolio_raw_return = (
            float(np.dot(weights, raw_values))
            if np.isfinite(raw_values).all()
            else None
        )
        for symbol, weight in zip(symbols, weights, strict=True):
            contribution = contributions.get(symbol, {}).get(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "weight": float(weight),
                    "contribution": contribution,
                    "portfolio_car": portfolio_car,
                    "portfolio_raw_return": portfolio_raw_return,
                    "scenario_median": summary["median"],
                    "scenario_min": summary.get("minimum", summary.get("min")),
                    "scenario_max": summary.get("maximum", summary.get("max")),
                    "events_used": summary.get("events_used", len(per_event)),
                    "data_pulled_at": data_pulled_at,
                    "data_through": data_through,
                }
            )
    return rows


def _raw_event_return(series, event_date, window):
    """Compound simple returns on the asset's own trading calendar."""
    available = series.dropna()
    day_zero = int(available.index.searchsorted(pd.Timestamp(event_date), side="left"))
    positions = np.arange(day_zero + window[0], day_zero + window[1] + 1)
    if positions[0] < 0 or positions[-1] >= len(available):
        return np.nan
    values = available.iloc[positions].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return np.nan
    return float(np.prod(1 + values) - 1)
