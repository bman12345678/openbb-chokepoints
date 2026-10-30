"""OpenBB router for the chokepoints extension (obb.chokepoints).

Commands here are thin wrappers: fetch data, call the pure functions in
``openbb_chokepoints.core``, and wrap the results in an OBBject.
"""

import warnings
from datetime import date
from typing import Annotated

from fastapi import Query
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
from openbb_chokepoints.models import EventStudyData

router = Router(prefix="", description="Chokepoint disruption risk analytics.")

DISCLAIMER = "Scenario analysis based on past events, not investment advice."


@router.command(
    methods=["GET"],
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
    warnings.warn(DISCLAIMER, OpenBBWarning, stacklevel=2)
    return OBBject(
        results=[EventStudyData.model_validate(r) for r in rows],
        provider="yfinance",
    )


def _split_symbols(symbols: str | list[str]) -> list[str]:
    items = [symbols] if isinstance(symbols, str) else symbols
    tickers = [s.strip().upper() for item in items for s in item.split(",")]
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        raise ValueError("Provide at least one symbol.")
    return tickers
