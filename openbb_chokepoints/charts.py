"""Build the optional charts described in the product/platform plan.

These views present event-study results; they do not fetch data or run the
market-model calculations. Charting imports stay lazy so the extra is optional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    plotted when sub-window rows are also present.
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
            figure.add_scatter(
                x=days,
                y=[None if value is None else value * 100 for value in cumulative],
                mode="lines+markers",
                name=symbol,
                line={"color": _COLORS[index % len(_COLORS)], "width": 2},
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
    figure.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_dark" if theme == "dark" else "plotly_white",
        xaxis_title="Trading day relative to event",
        yaxis_title="Cumulative abnormal return (%)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 70, "r": 25, "t": 85, "b": 65},
        annotations=(
            [
                {
                    "text": "Illustrative synthetic data; not market observations.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": -0.2,
                    "showarrow": False,
                    "font": {"size": 10},
                }
            ]
            if sample_mode
            else None
        ),
    )
    return figure


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
