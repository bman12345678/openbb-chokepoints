"""Deterministic chart behavior checks for the product chart set."""

import pytest

pytest.importorskip("openbb_charting")

from openbb_chokepoints.charts import (
    events_chart,
    exposure_chart,
    monitor_chart,
    optimize_chart,
    stress_test_chart,
)


def test_monitor_draws_traffic_band_flags_and_source_dates():
    figure = monitor_chart(
        [
            {
                "chokepoint": "suez_canal",
                "vessel_type": "container",
                "as_of": "2024-01-01",
                "avg_7d": 80,
                "baseline_median": 100,
                "normal_low": 70,
                "normal_high": 130,
                "flagged": False,
                "data_through": "2024-01-01",
                "data_pulled_at": "2024-01-03T10:00:00Z",
            },
            {
                "chokepoint": "suez_canal",
                "vessel_type": "container",
                "as_of": "2024-01-02",
                "avg_7d": 55,
                "baseline_median": 100,
                "normal_low": 70,
                "normal_high": 130,
                "flagged": True,
            },
        ]
    )

    assert any((trace.name or "").endswith("flagged") for trace in figure.data)
    assert any(trace.fill == "tonexty" for trace in figure.data)
    assert "Vessels per day" in figure.layout.yaxis.title.text
    assert "Data through 2024-01-01" in figure.layout.title.text


def test_events_distinguish_recovery_markers_and_event_type():
    figure = events_chart(
        [
            {
                "event_id": "attack",
                "chokepoint": "bab_el_mandeb_strait",
                "event_type": "conflict",
                "direction": "disruption",
                "news_date": "2023-12-15",
                "description": "Carrier pause",
            },
            {
                "event_id": "ceasefire",
                "chokepoint": "bab_el_mandeb_strait",
                "event_type": "resolution",
                "direction": "recovery",
                "news_date": "2025-01-15",
                "description": "Ceasefire",
            },
        ]
    )

    traces = {trace.name: trace for trace in figure.data}
    assert set(traces) == {"conflict", "resolution"}
    assert traces["conflict"].marker.symbol[0] == "circle"
    assert traces["resolution"].marker.symbol[0] == "circle-open"


def test_exposure_uses_percent_axis_and_lists_ranked_industries():
    figure = exposure_chart(
        [
            {
                "chokepoint": "suez_canal",
                "category": "vessel_type",
                "name": "container",
                "share_count": 0.32,
            },
            {
                "chokepoint": "suez_canal",
                "category": "industry",
                "name": "Manufacturing",
                "rank": 1,
            },
        ]
    )

    assert list(figure.data[0].x) == [32.0]
    assert "Manufacturing" in figure.layout.annotations[0].text
    assert "(%)" in figure.layout.xaxis.title.text


def test_stress_test_shows_holding_contributions_and_event_outcomes():
    figure = stress_test_chart(
        [
            {
                "event_id": "event_a",
                "symbol": "ZIM",
                "contribution": -0.03,
                "portfolio_car": -0.04,
                "scenario_median": -0.02,
                "events_used": 2,
            },
            {
                "event_id": "event_a",
                "symbol": "SPY",
                "contribution": 0.01,
                "portfolio_car": -0.04,
                "scenario_median": -0.02,
                "events_used": 2,
            },
            {
                "event_id": "event_b",
                "symbol": "ZIM",
                "contribution": -0.01,
                "portfolio_car": 0.01,
                "scenario_median": -0.02,
                "events_used": 2,
            },
            {
                "event_id": "event_b",
                "symbol": "SPY",
                "contribution": 0.01,
                "portfolio_car": 0.01,
                "scenario_median": -0.02,
                "events_used": 2,
            },
        ]
    )

    assert any(trace.type == "bar" for trace in figure.data)
    assert any(trace.type == "scatter" for trace in figure.data)
    assert list(figure.data[0].x) == pytest.approx([-2.0, 1.0])
    assert len(figure.data[1].x) == 2
    assert any(shape.type == "line" for shape in figure.layout.shapes)
    assert "2 events" in figure.layout.title.text
    assert "need not sum" in figure.layout.title.text


def test_optimizer_compares_original_and_proposed_weights():
    figure = optimize_chart(
        [
            {
                "symbol": "ZIM",
                "current_weight": 0.5,
                "new_weight": 0.3,
                "loss_before": -0.06,
                "loss_after": -0.03,
                "turnover": 0.2,
            },
            {
                "symbol": "SPY",
                "current_weight": 0.5,
                "new_weight": 0.7,
                "loss_before": -0.06,
                "loss_after": -0.03,
                "turnover": 0.2,
            },
        ]
    )

    assert len(figure.data) == 2
    assert list(figure.data[0].y) == [50.0, 50.0]
    assert list(figure.data[1].y) == [30.0, 70.0]
    assert "6.0%" in figure.layout.title.text
    assert "Turnover 20.0%" in figure.layout.title.text
