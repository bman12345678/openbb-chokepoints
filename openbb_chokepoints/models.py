"""Pydantic output models for obb.chokepoints commands."""

from datetime import date

from openbb_core.provider.abstract.data import Data
from pydantic import Field

_PERCENT = {"x-unit_measurement": "percent", "x-frontend_multiply": 100}


class EventStudyData(Data):
    """Market-model event study result for one symbol and one CAR window."""

    symbol: str = Field(title="Symbol", description="Ticker of the asset.")
    event_date: date = Field(
        title="Event Date", description="Calendar date of the event as given."
    )
    day0: date = Field(
        title="Day 0",
        description="First trading day of the symbol on or after the event date.",
    )
    window: str = Field(
        title="CAR Window",
        description="Inclusive trading-day window around day 0, e.g. '(-5, 20)'.",
    )
    car: float = Field(
        title="CAR",
        description="Cumulative abnormal return over the window, as a decimal.",
        json_schema_extra=_PERCENT,
    )
    car_t: float | None = Field(
        default=None,
        title="CAR t-stat",
        description="CAR divided by its standard error, including the error from"
        " estimating alpha and beta (MacKinlay 1997).",
    )
    p_value: float | None = Field(
        default=None,
        title="p-value",
        description="Two-sided p-value of the CAR t-stat from the standard normal.",
    )
    alpha: float = Field(
        title="Alpha",
        description="Market-model intercept (daily return, decimal) from the"
        " estimation window.",
        json_schema_extra=_PERCENT,
    )
    beta: float = Field(
        title="Beta",
        description="Market-model slope on the market return from the estimation"
        " window.",
    )
    sigma_eps: float = Field(
        title="Residual Volatility",
        description="Standard deviation of the daily market-model residuals"
        " (decimal), with L1 - 2 degrees of freedom.",
        json_schema_extra=_PERCENT,
    )
    n_est: int = Field(
        title="Estimation Observations",
        description="Number of trading days used to fit the market model (L1).",
    )
    ar_by_day: list[float | None] = Field(
        title="Abnormal Returns by Day",
        description="Daily abnormal returns (decimals) for every day of the full"
        " event window, in order; null where data is missing.",
        json_schema_extra=_PERCENT,
    )
