"""Build the optional charts described in the product/platform plan.

These views present event-study results; they do not fetch data or run the
market-model calculations. Charting imports stay lazy so the extra is optional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

# Reuse this palette across symbol traces in the event-study view.
_COLORS = (
    "#2E86AB",
    "#F18F01",
    "#A23B72",
    "#3B8C6E",
    "#C73E1D",
    "#6F63A6",
    "#4B6B8A",
    "#9A6B45",
)


def event_study_chart(
    rows: Sequence[Any],
    event_window: tuple[int, int] = (-5, 20),
    *,
    sample_mode: bool = False,
    theme: str = "dark",
):
    """Plot the CAR path from the event-study command's result rows.

    ``ar_by_day`` follows the event-study result model. Summing abnormal
    returns by day gives the CAR path (MacKinlay, 1997); values are decimals
    and the chart displays percentages. Only the full requested window is
    plotted when sub-window rows are also present. If a row also supplies
    ``car_var_by_day`` (decimal-squared) or ``car_se_by_day`` (decimal), a
    pointwise 95% confidence band is drawn. These values are calculated by
    the analysis layer so the chart does not duplicate its variance method.
    """
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = [_as_record(row) for row in rows]
    records = [row for row in records if row]
    full_window = _window_label(event_window)
    selected: dict[str, dict[str, Any]] = {}

    for row in records:
        symbol = str(row.get("symbol", "")).strip()
        values = row.get("ar_by_day")
        if not symbol or not isinstance(values, (list, tuple)):
            continue
        if row.get("window") == full_window or symbol not in selected:
            selected[symbol] = row

    figure = OpenBBFigure()
    if not selected:
        figure.add_annotation(
            text="No event-study abnormal-return rows to display.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    else:
        start = int(event_window[0])
        for index, (symbol, row) in enumerate(sorted(selected.items())):
            values = row.get("ar_by_day", [])
            cumulative = _cumulative_with_gaps(values)
            days = list(range(start, start + len(values)))
            bounds = _confidence_bounds(row, len(values))
            color = _COLORS[index % len(_COLORS)]
            if bounds is not None:
                for path in _confidence_band_paths(days, cumulative, bounds):
                    figure.add_shape(
                        type="path",
                        path=path,
                        xref="x",
                        yref="y",
                        fillcolor=_with_alpha(color, 0.18),
                        line={"color": _with_alpha(color, 0), "width": 0},
                        layer="below",
                    )
            figure.add_scatter(
                x=days,
                y=[None if value is None else value * 100 for value in cumulative],
                mode="lines+markers",
                name=symbol,
                line={"color": color, "width": 2},
                marker={"size": 5},
                connectgaps=False,
                customdata=values,
                hovertemplate=(
                    "Trading day %{x}<br>CAR path: %{y:.2f}%"
                    "<br>Daily AR: %{customdata:.2%}<extra>%{fullData.name}</extra>"
                ),
            )

        figure.add_vline(
            x=0,
            line={"color": "#F5C16C", "dash": "dash", "width": 2},
            annotation_text="Day 0",
            annotation_position="top",
        )

    title = "Cumulative Abnormal Returns"
    if sample_mode:
        title += " — synthetic sample"
    has_confidence_band = any(
        _confidence_bounds(row, len(row.get("ar_by_day", []))) is not None
        for row in selected.values()
    )
    if has_confidence_band:
        title += " (shaded area: 95% confidence band)"
    figure.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_dark" if theme == "dark" else "plotly_white",
        xaxis_title="Trading day relative to event",
        yaxis_title="Cumulative abnormal return (%)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 70, "r": 25, "t": 85, "b": 90},
        annotations=[
            {
                "text": (
                    "Scenario analysis based on past events; not investment advice."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.18,
                "showarrow": False,
                "font": {"size": 10},
            },
            *(
                [
                    {
                        "text": "Illustrative synthetic data; not market observations.",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": -0.25,
                        "showarrow": False,
                        "font": {"size": 10},
                    }
                ]
                if sample_mode
                else []
            ),
        ],
    )
    return figure


def monitor_chart(rows: Sequence[Any], *, theme: str = "dark"):
    """Plot daily traffic, a normal band, and persistent flagged observations."""
    import plotly.graph_objects as go
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = _records(rows)
    figure = OpenBBFigure()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row.get("chokepoint", "")), str(row.get("vessel_type", "")))
        groups.setdefault(key, []).append(row)
    for (chokepoint, vessel_type), group in sorted(groups.items()):
        group.sort(key=lambda row: str(row.get("as_of", "")))
        dates = [row.get("as_of") for row in group]
        values = [_number(row.get("avg_7d")) for row in group]
        name = f"{chokepoint} · {vessel_type}"
        figure.add_trace(go.Scatter(x=dates, y=values, mode="lines", name=name))
        lower = [_number(row.get("normal_low")) for row in group]
        upper = [_number(row.get("normal_high")) for row in group]
        if any(value is not None for value in lower + upper):
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    name=f"{name} normal range",
                    opacity=0.18,
                )
            )
        else:
            baseline = [_number(row.get("baseline_median")) for row in group]
            figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=baseline,
                    mode="lines",
                    line={"dash": "dot"},
                    name=f"{name} baseline",
                )
            )
        flagged = [row for row in group if row.get("flagged")]
        if flagged:
            figure.add_trace(
                go.Scatter(
                    x=[row.get("as_of") for row in flagged],
                    y=[_number(row.get("avg_7d")) for row in flagged],
                    mode="markers",
                    name=f"{name} flagged",
                    marker={"symbol": "x", "size": 10},
                )
            )
    pulled = _latest(records, "data_pulled_at")
    through = _latest(records, "data_through")
    _finish_chart(
        figure,
        "Chokepoint Traffic Monitor",
        "Date",
        "Vessels per day (7-day average)",
        theme,
        subtitle=_date_note(pulled, through),
    )
    return figure


def events_chart(rows: Sequence[Any], *, theme: str = "dark"):
    """Show catalog events on a chokepoint timeline; recovery markers are hollow."""
    import plotly.graph_objects as go
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = _records(rows)
    figure = OpenBBFigure()
    colors = {
        "conflict": "#C73E1D",
        "blockage": "#F18F01",
        "climate": "#2E86AB",
        "policy": "#A23B72",
        "resolution": "#3B8C6E",
        "detected": "#6F63A6",
    }
    for event_type in sorted(
        {str(row.get("event_type", "unknown")) for row in records}
    ):
        group = [
            row
            for row in records
            if str(row.get("event_type", "unknown")) == event_type
        ]
        symbols = [
            "circle-open"
            if str(row.get("direction", "disruption")) == "recovery"
            else "circle"
            for row in group
        ]
        figure.add_trace(
            go.Scatter(
                x=[row.get("news_date") for row in group],
                y=[row.get("chokepoint") for row in group],
                mode="markers",
                name=event_type,
                text=[row.get("description", "") for row in group],
                customdata=[
                    [row.get("event_id"), row.get("direction"), row.get("confidence")]
                    for row in group
                ],
                marker={
                    "color": colors.get(event_type, "#6F63A6"),
                    "symbol": symbols,
                    "size": 12,
                },
                hovertemplate=(
                    "%{y}<br>%{x|%Y-%m-%d}<br>%{text}<br>"
                    "%{customdata[0]} · %{customdata[1]} · %{customdata[2]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    _finish_chart(
        figure,
        "Chokepoint Event Timeline",
        "News date",
        "Chokepoint",
        theme,
        subtitle=(
            "Recovery events use hollow markers. "
            "Scenario analysis based on past events."
        ),
    )
    return figure


def exposure_chart(
    rows: Sequence[Any], *, measure: str = "share_count", theme: str = "dark"
):
    """Compare vessel exposure shares and list ranked PortWatch industries."""
    import plotly.graph_objects as go
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = _records(rows)
    figure = OpenBBFigure()
    shares = [row for row in records if _number(row.get(measure)) is not None]
    labels = [f"{row.get('name', '')} · {row.get('chokepoint', '')}" for row in shares]
    figure.add_trace(
        go.Bar(
            x=[_number(row.get(measure)) * 100 for row in shares],
            y=labels,
            orientation="h",
            name="Vessel share (%)",
        )
    )
    ranked = [row for row in records if row.get("rank") is not None]
    if ranked:
        lines = [
            f"{row.get('chokepoint')}: #{row.get('rank')} {row.get('name')}"
            for row in sorted(
                ranked, key=lambda r: (str(r.get("chokepoint")), int(r.get("rank", 99)))
            )
        ]
        figure.add_annotation(
            text="<br>".join(lines),
            xref="paper",
            yref="paper",
            x=1.02,
            y=1,
            showarrow=False,
            xanchor="left",
            align="left",
        )
    _finish_chart(
        figure,
        "Chokepoint Exposure",
        "Share of traffic (%)",
        "Exposure category",
        theme,
        subtitle=(
            "PortWatch industry ranks, not firm-level shares. "
            "Scenario analysis based on historical traffic."
        ),
    )
    return figure


def stress_test_chart(rows: Sequence[Any], *, theme: str = "dark"):
    """Show per-holding contributions and the portfolio result for each event."""
    import plotly.graph_objects as go
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = _records(rows)
    figure = OpenBBFigure()
    contributions: dict[str, list[float]] = {}
    for row in records:
        symbol, value = row.get("symbol"), _number(row.get("contribution"))
        if symbol and value is not None:
            contributions.setdefault(str(symbol), []).append(value)
    if contributions:
        median_contributions = {
            symbol: median(values) for symbol, values in contributions.items()
        }
        ordered = sorted(median_contributions.items(), key=lambda item: item[1])
        figure.add_trace(
            go.Bar(
                x=[value * 100 for _, value in ordered],
                y=[symbol for symbol, _ in ordered],
                orientation="h",
                name="Median contribution (%)",
            )
        )
    event_rows = []
    seen_events = set()
    for row in records:
        event_id = row.get("event_id")
        if row.get("portfolio_car") is None or event_id in seen_events:
            continue
        event_rows.append(row)
        seen_events.add(event_id)
    if event_rows:
        figure.add_trace(
            go.Scatter(
                x=[_number(row.get("portfolio_car")) * 100 for row in event_rows],
                y=[row.get("event_id") for row in event_rows],
                mode="markers",
                name="Historical portfolio CAR (%)",
                marker={"symbol": "circle", "size": 10},
            )
        )
    median_value = next(
        (
            _number(row.get("scenario_median"))
            for row in records
            if row.get("scenario_median") is not None
        ),
        None,
    )
    if median_value is not None:
        figure.add_vline(
            x=median_value * 100,
            line={"dash": "dash", "color": "#F5C16C"},
            annotation_text=f"Median {median_value:.1%}",
        )
    count = next(
        (row.get("events_used") for row in records if row.get("events_used")), 0
    )
    _finish_chart(
        figure,
        f"Historical Portfolio Stress Test · {count} events",
        "Abnormal return / contribution (%)",
        "Holding or event",
        theme,
        subtitle=(
            "Scenario analysis based on past events; holding contributions "
            "need not sum to the portfolio median."
        ),
    )
    return figure


def optimize_chart(rows: Sequence[Any], *, theme: str = "dark"):
    """Compare current and proposed weights and report loss/turnover context."""
    import plotly.graph_objects as go
    from openbb_charting.core.openbb_figure import OpenBBFigure

    records = _records(rows)
    figure = OpenBBFigure()
    labels = [str(row.get("symbol", "")) for row in records]
    current = [(_number(row.get("current_weight")) or 0) * 100 for row in records]
    proposed = [(_number(row.get("new_weight")) or 0) * 100 for row in records]
    figure.add_trace(go.Bar(x=labels, y=current, name="Current weight (%)"))
    figure.add_trace(go.Bar(x=labels, y=proposed, name="Proposed weight (%)"))
    first = records[0] if records else {}
    before, after = _number(first.get("loss_before")), _number(first.get("loss_after"))
    turnover = _number(first.get("turnover"))
    detail = []
    if before is not None and after is not None:
        detail.append(f"Estimated loss {before:.1%} → {after:.1%}")
    if turnover is not None:
        detail.append(f"Turnover {turnover:.1%}")
    _finish_chart(
        figure,
        "Exposure-Limited Portfolio Weights",
        "Holding",
        "Portfolio weight (%)",
        theme,
        subtitle=" · ".join(detail) or "Scenario analysis based on past events.",
    )
    figure.update_layout(barmode="group")
    return figure


def _records(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [record for row in rows if (record := _as_record(row))]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _latest(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return max(values, key=str) if values else None


def _date_note(pulled: Any, through: Any) -> str:
    parts = []
    if through is not None:
        parts.append(f"Data through {through}")
    if pulled is not None:
        parts.append(f"pulled {pulled}")
    parts.append("Scenario analysis; not investment advice")
    return " · ".join(parts)


def _finish_chart(
    figure: Any, title: str, x_title: str, y_title: str, theme: str, *, subtitle: str
):
    figure.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5},
        template="plotly_dark" if theme == "dark" else "plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin={
            "l": 100,
            "r": 240 if title == "Chokepoint Exposure" else 35,
            "t": 105,
            "b": 65,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )


def _as_record(row: Any) -> dict[str, Any]:
    """Convert a result model or mapping to the chart's row format."""
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json", exclude_none=False)
    return {}


def _window_label(window: tuple[int, int]) -> str:
    """Format window bounds like the event-study output model does."""
    return f"({int(window[0])}, {int(window[1])})"


def _cumulative_with_gaps(values: Sequence[Any]) -> list[float | None]:
    """Accumulate daily abnormal returns while keeping missing days as gaps."""
    running = 0.0
    result: list[float | None] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        number = float(value)
        if number != number:  # NaN should remain a gap in the chart.
            result.append(None)
            continue
        running += number
        result.append(running)
    return result


def _confidence_bounds(
    row: Mapping[str, Any], length: int
) -> list[tuple[float, float] | None] | None:
    """Read pointwise CAR variance or standard error from an output row."""
    variances = row.get("car_var_by_day")
    standard_errors = row.get("car_se_by_day")
    if isinstance(variances, (list, tuple)) and len(variances) == length:
        result = []
        for value in variances:
            number = _finite_nonnegative(value)
            result.append(None if number is None else 1.96 * number**0.5 * 100)
        return [(value, value) if value is not None else None for value in result]
    if isinstance(standard_errors, (list, tuple)) and len(standard_errors) == length:
        result = []
        for value in standard_errors:
            number = _finite_nonnegative(value)
            result.append(None if number is None else 1.96 * number * 100)
        return [(value, value) if value is not None else None for value in result]
    return None


def _finite_nonnegative(value: Any) -> float | None:
    """Return a finite nonnegative number, or None for unusable data."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _confidence_band_paths(
    days: Sequence[int],
    cumulative: Sequence[float | None],
    margins: Sequence[tuple[float, float] | None],
) -> list[str]:
    """Build closed polygon paths for contiguous portions of the confidence band."""
    paths: list[str] = []
    upper: list[tuple[int, float]] = []
    lower: list[tuple[int, float]] = []

    def flush() -> None:
        if not upper:
            return
        points = upper + list(reversed(lower))
        paths.append(
            " ".join(
                [f"M {points[0][0]},{points[0][1]:.12g}"]
                + [f"L {x},{y:.12g}" for x, y in points[1:]]
                + ["Z"]
            )
        )
        upper.clear()
        lower.clear()

    for day, value, margin in zip(days, cumulative, margins, strict=True):
        if value is None or margin is None:
            flush()
            continue
        low_margin, high_margin = margin
        upper.append((day, value + high_margin))
        lower.append((day, value - low_margin))
    flush()
    return paths


def _with_alpha(color: str, alpha: float) -> str:
    """Convert the chart's hex color to an rgba color."""
    channels = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({channels[0]},{channels[1]},{channels[2]},{alpha})"
