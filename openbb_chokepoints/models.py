"""Pydantic output models for obb.chokepoints commands."""

from datetime import date, datetime
from typing import Literal

from openbb_core.provider.abstract.data import Data
from pydantic import BaseModel, Field

_PERCENT = {"x-unit_measurement": "percent", "x-frontend_multiply": 100}


class PortfolioHolding(BaseModel):
    """One user supplied holding in a portfolio request."""

    symbol: str = Field(description="Tradable asset symbol.")
    weight: float = Field(ge=0, description="Portfolio weight as a decimal.")


class StressTestParameters(BaseModel):
    """POST body for historical scenario stress testing."""

    data: list[PortfolioHolding]
    chokepoint: str
    event_type: str
    direction: Literal["disruption", "recovery"] = "disruption"
    window: tuple[int, int] = (0, 20)


class OptimizeParameters(BaseModel):
    """POST body for scenario-constrained portfolio optimization."""

    data: list[PortfolioHolding]
    chokepoint: str
    event_type: str
    max_scenario_loss: float = Field(gt=0, le=1)
    objective: Literal["min_change", "min_variance"] = "min_change"
    constraint: Literal["average", "every_event"] = "average"
    direction: Literal["disruption", "recovery"] = "disruption"


class EventStudyData(Data):
    """Market-model event study result for one symbol and one CAR window."""

    symbol: str = Field(title="Symbol", description="Ticker of the asset.")
    event_date: date = Field(
        title="Event Date", description="Calendar date of the event as given."
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time the market-data provider was queried.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest market observation available to the calculation.",
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
    car_var: float | None = Field(
        default=None,
        title="CAR Variance",
        description=(
            "Variance of the cumulative abnormal return in decimal-squared units."
        ),
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
    car_var_by_day: list[float | None] | None = Field(
        default=None,
        title="CAR Variance by Day",
        description=(
            "Variance of the cumulative abnormal-return path at each relative "
            "event-window day, in decimal-squared units. Null where unavailable."
        ),
    )


class MonitorData(Data):
    """Traffic-monitor observations, one row per date/chokepoint/vessel type."""

    chokepoint: str = Field(title="Chokepoint", description="PortWatch chokepoint key.")
    vessel_type: str = Field(
        title="Vessel Type", description="PortWatch vessel category."
    )
    as_of: date = Field(
        title="Date", description="Date represented by this observation."
    )
    avg_7d: float | None = Field(
        title="7-Day Average", description="Seven-day vessel average."
    )
    baseline_median: float | None = Field(
        title="Baseline Median", description="Median of prior seven-day averages."
    )
    normal_low: float | None = Field(
        default=None,
        title="Normal Range Low",
        description="Lower robust-normal traffic bound.",
    )
    normal_high: float | None = Field(
        default=None,
        title="Normal Range High",
        description="Upper robust-normal traffic bound.",
    )
    robust_z: float | None = Field(
        title="Robust Z-Score", description="MAD-scaled deviation."
    )
    pct_vs_baseline: float | None = Field(
        title="Change vs Baseline",
        description="Fractional change against baseline.",
        json_schema_extra=_PERCENT,
    )
    pct_vs_last_year: float | None = Field(
        default=None,
        title="Change vs Prior Year",
        description="Fractional change vs 52 weeks earlier.",
        json_schema_extra=_PERCENT,
    )
    flagged: bool = Field(
        title="Flagged", description="Persistent abnormal traffic signal."
    )
    low_traffic: bool = Field(
        default=False,
        title="Low Traffic",
        description="Low-volume series; interpret robust scores cautiously.",
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time the source was queried.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest source observation date.",
    )


class EventCatalogData(Data):
    """Catalog or automatically detected chokepoint event."""

    event_id: str = Field(title="Event ID", description="Stable catalog identifier.")
    news_time_utc: datetime | None = Field(
        default=None,
        title="News Time (UTC)",
        description="Source time when known; supports exchange-local event alignment.",
    )
    chokepoint: str = Field(title="Chokepoint", description="PortWatch chokepoint key.")
    event_type: str = Field(
        title="Event Type",
        description="Conflict, blockage, climate, policy, resolution, or detected.",
    )
    direction: str = Field(title="Direction", description="Disruption or recovery.")
    news_date: date = Field(
        title="News Date",
        description="First public report; event-study day zero anchor.",
    )
    end_date: date | None = Field(
        default=None, title="End Date", description="Known end date, if available."
    )
    confidence: str = Field(title="Confidence", description="Catalog confidence level.")
    description: str = Field(title="Description", description="Short event summary.")
    sources: list[str] = Field(
        default_factory=list,
        title="Sources",
        description="Source links supporting the event date.",
    )
    related_chokepoints: list[str] = Field(
        default_factory=list,
        title="Related Chokepoints",
        description="Other affected passages.",
    )
    overlaps_with: list[str] = Field(
        default_factory=list,
        title="Overlapping Events",
        description="Related shocks for clustered analysis.",
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time the catalog was read.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest event observation represented.",
    )


class ExposureData(Data):
    """Vessel or industry exposure and associated market proxies."""

    chokepoint: str = Field(title="Chokepoint", description="PortWatch chokepoint key.")
    category: str = Field(
        title="Category",
        description="Vessel type, PortWatch industry, or chokepoint-linked transport.",
    )
    name: str = Field(
        title="Name",
        description="Vessel type, PortWatch industry, or mapped transport group.",
    )
    share_count: float | None = Field(
        default=None,
        title="Count Share",
        description="Share of vessel counts.",
        json_schema_extra=_PERCENT,
    )
    share_capacity: float | None = Field(
        default=None,
        title="Capacity Share",
        description="Share of vessel capacity in metric tons.",
        json_schema_extra=_PERCENT,
    )
    rank: int | None = Field(
        default=None,
        title="Industry Rank",
        description="PortWatch industry rank when available.",
    )
    proxy_symbols: list[str] = Field(
        default_factory=list,
        title="Market Proxies",
        description="Tradable symbols associated with this exposure.",
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time the source was queried.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest source observation date.",
    )


class FirmExposureData(Data):
    """Mapped relationship between a market symbol and a chokepoint."""

    symbol: str = Field(title="Symbol", description="Market symbol being searched.")
    chokepoint: str = Field(title="Chokepoint", description="Related passage key.")
    relationship: str = Field(
        title="Relationship",
        description="Route-linked company or sector proxy relationship.",
    )
    sector: str | None = Field(
        default=None,
        title="Sector",
        description="Mapped sector for sector proxy relationships.",
    )
    sector_score: float | None = Field(
        default=None,
        title="Ordinal Sector Score",
        description=(
            "Rank-weighted PortWatch industry score; not a measured share of trade."
        ),
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time the PortWatch and proxy mappings were read.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Date represented by PortWatch's exposure snapshot.",
    )


class StressTestData(Data):
    """Per-event portfolio results and per-holding contributions."""

    event_id: str = Field(title="Event ID", description="Comparable historical event.")
    symbol: str | None = Field(
        default=None,
        title="Symbol",
        description="Holding; null on portfolio summary rows.",
    )
    weight: float | None = Field(
        default=None,
        title="Weight",
        description="Portfolio weight as a decimal.",
        json_schema_extra=_PERCENT,
    )
    contribution: float | None = Field(
        default=None,
        title="Contribution",
        description="Weight times holding CAR; decimal return.",
        json_schema_extra=_PERCENT,
    )
    portfolio_car: float | None = Field(
        default=None,
        title="Portfolio CAR",
        description="Portfolio abnormal return for this event.",
        json_schema_extra=_PERCENT,
    )
    portfolio_raw_return: float | None = Field(
        default=None,
        title="Portfolio Raw Return",
        description="Raw return over the same window.",
        json_schema_extra=_PERCENT,
    )
    scenario_median: float | None = Field(
        default=None,
        title="Scenario Median",
        description="Median portfolio CAR across comparable events.",
        json_schema_extra=_PERCENT,
    )
    scenario_min: float | None = Field(
        default=None,
        title="Scenario Minimum",
        description="Minimum portfolio CAR across comparable events.",
        json_schema_extra=_PERCENT,
    )
    scenario_max: float | None = Field(
        default=None,
        title="Scenario Maximum",
        description="Maximum portfolio CAR across comparable events.",
        json_schema_extra=_PERCENT,
    )
    events_used: int = Field(
        default=0,
        title="Events Used",
        description="Comparable historical events used in this result.",
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time market data was queried.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest market observation used.",
    )


class OptimizeData(Data):
    """Optimized portfolio weights and scenario summary."""

    symbol: str = Field(title="Symbol", description="Portfolio holding or hedge.")
    current_weight: float = Field(
        title="Current Weight",
        description="Original portfolio weight.",
        json_schema_extra=_PERCENT,
    )
    new_weight: float = Field(
        title="New Weight",
        description="Proposed portfolio weight or hedge size.",
        json_schema_extra=_PERCENT,
    )
    change: float = Field(
        title="Weight Change",
        description="New minus current weight.",
        json_schema_extra=_PERCENT,
    )
    loss_before: float | None = Field(
        default=None,
        title="Loss Before",
        description="Scenario loss before optimization.",
        json_schema_extra=_PERCENT,
    )
    loss_after: float | None = Field(
        default=None,
        title="Loss After",
        description="Scenario loss after optimization.",
        json_schema_extra=_PERCENT,
    )
    loss_min_before: float | None = Field(
        default=None,
        title="Loss Range Low Before",
        description="Lowest historical scenario return before optimization.",
        json_schema_extra=_PERCENT,
    )
    loss_max_before: float | None = Field(
        default=None,
        title="Loss Range High Before",
        description="Highest historical scenario return before optimization.",
        json_schema_extra=_PERCENT,
    )
    loss_min_after: float | None = Field(
        default=None,
        title="Loss Range Low After",
        description="Lowest historical scenario return after optimization.",
        json_schema_extra=_PERCENT,
    )
    loss_max_after: float | None = Field(
        default=None,
        title="Loss Range High After",
        description="Highest historical scenario return after optimization.",
        json_schema_extra=_PERCENT,
    )
    turnover: float | None = Field(
        default=None,
        title="Turnover",
        description="Half the sum of absolute weight changes.",
        json_schema_extra=_PERCENT,
    )
    tracking_error: float | None = Field(
        default=None,
        title="Tracking Error",
        description="Annualized tracking error versus the original portfolio.",
        json_schema_extra=_PERCENT,
    )
    objective: str = Field(
        default="min_change",
        title="Objective",
        description="Optimization objective used.",
    )
    constraint: str = Field(
        default="average",
        title="Constraint",
        description="Average scenario or every-event loss constraint.",
    )
    data_pulled_at: datetime | None = Field(
        default=None,
        title="Data Pulled At",
        description="UTC time market data was queried.",
    )
    data_through: date | None = Field(
        default=None,
        title="Data Through",
        description="Latest market observation used.",
    )
