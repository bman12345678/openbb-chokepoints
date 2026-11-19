"""Route schemas and command adapters for Sahya-owned product wiring."""

import sys
import types
from datetime import date

import pandas as pd
import pytest

pytest.importorskip("openbb_core")

from openbb_chokepoints import router as router_module
from openbb_chokepoints.models import (
    EventCatalogData,
    EventStudyData,
    ExposureData,
    MonitorData,
    OptimizeData,
    OptimizeParameters,
    PortfolioHolding,
    StressTestData,
    StressTestParameters,
)


def test_all_planned_command_routes_register_and_have_documentation():
    routes = {route.path: route for route in router_module.router.api_router.routes}

    assert {
        "/monitor",
        "/events",
        "/exposure",
        "/event_study",
        "/stress_test",
        "/optimize",
    } <= routes.keys()
    for name in (
        "monitor",
        "events",
        "exposure",
        "event_study",
        "stress_test",
        "optimize",
    ):
        route = routes[f"/{name}"]
        assert route.openapi_extra.get("widget_config")
        assert route.openapi_extra.get("mcp_config")


def test_output_models_describe_percent_units_and_scenario_rows():
    assert (
        EventStudyData.model_fields["car"].json_schema_extra["x-frontend_multiply"]
        == 100
    )
    assert (
        MonitorData.model_fields["pct_vs_baseline"].json_schema_extra[
            "x-unit_measurement"
        ]
        == "percent"
    )
    assert EventCatalogData.model_fields["news_date"].annotation is date
    assert ExposureData.model_fields["rank"].default is None
    assert StressTestData.model_fields["events_used"].default == 0
    assert (
        OptimizeData.model_fields["turnover"].json_schema_extra["x-frontend_multiply"]
        == 100
    )


def test_exposure_wrapper_adapts_portwatch_fields(monkeypatch):
    fake = types.ModuleType("openbb_chokepoints.data.portwatch")
    fake.get_chokepoint_info = lambda: pd.DataFrame(
        [
            {
                "name": "suez_canal",
                "vessel_count_total": 100,
                "vessel_count_tanker": 20,
                "vessel_count_container": 30,
                "vessel_count_dry_bulk": 25,
                "vessel_count_general_cargo": 15,
                "vessel_count_roro": 10,
                "industry_top1": "Manufacturing",
                "industry_top2": "Energy",
                "industry_top3": None,
            }
        ]
    )
    fake.get_chokepoint_volume = lambda *args: pd.DataFrame(
        {
            "vessel_type": ["tanker", "container"],
            "vessels": [2.0, 8.0],
            "capacity_tons": [20.0, 80.0],
        }
    )
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    result = router_module.exposure("suez_canal")

    vessel_rows = [row for row in result.results if row.category == "vessel_type"]
    industry_rows = [row for row in result.results if row.category == "industry"]
    assert sum(row.share_count for row in vessel_rows) == pytest.approx(1.0)
    assert [(row.name, row.rank) for row in industry_rows] == [
        ("Manufacturing", 1),
        ("Energy", 2),
    ]


def test_monitor_wrapper_adapts_long_portwatch_rows(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    traffic = pd.DataFrame(
        {
            "date": dates,
            "chokepoint": "suez_canal",
            "vessel_type": "container",
            "vessels": [100.0] * 170 + [50.0] * 10,
            "capacity_tons": [1000.0] * 180,
        }
    )
    traffic.attrs["data_pulled_at"] = "2024-07-01T12:00:00Z"
    traffic.attrs["data_through"] = "2024-06-28"
    portwatch = types.ModuleType("openbb_chokepoints.data.portwatch")
    portwatch.get_chokepoint_volume = lambda *args: traffic.copy()
    detection = types.ModuleType("openbb_chokepoints.core.detection")
    detection.robust_z = lambda series, **kwargs: pd.Series(0.0, index=series.index)
    detection.flag = lambda scores, **kwargs: pd.Series(False, index=scores.index)
    monkeypatch.setitem(sys.modules, portwatch.__name__, portwatch)
    monkeypatch.setitem(sys.modules, detection.__name__, detection)

    result = router_module.monitor(chokepoint="suez_canal", lookback_days=90)

    assert result.results
    assert result.results[-1].chokepoint == "suez_canal"
    assert result.results[-1].normal_low is not None
    assert result.results[-1].data_through.isoformat() == "2024-06-28"


def _mock_event_returns(monkeypatch):
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    returns = pd.DataFrame({"ZIM": 0.001, "SPY": 0.0005}, index=dates)
    events = [
        {
            "event_id": "test_conflict",
            "news_date": "2023-12-01",
            "chokepoint": "bab_el_mandeb_strait",
        }
    ]
    event_module = types.ModuleType("openbb_chokepoints.data.events")
    event_module.load_events = lambda **kwargs: events
    monkeypatch.setitem(sys.modules, event_module.__name__, event_module)
    monkeypatch.setattr(router_module, "get_returns", lambda *args: returns)
    monkeypatch.setattr(
        router_module,
        "run_event_study",
        lambda **kwargs: [
            {
                "symbol": symbol,
                "window": "(0, 20)",
                "car": -0.04 if symbol == "ZIM" else 0.0,
            }
            for symbol in kwargs["returns_df"].columns
        ],
    )


def test_stress_test_wrapper_calls_teammate_core_and_builds_result_rows(monkeypatch):
    _mock_event_returns(monkeypatch)
    core = types.ModuleType("openbb_chokepoints.core.stress")
    core.stress_test = lambda weights, matrix: {
        "per_event": {"test_conflict": -0.04},
        "summary": {
            "median": -0.04,
            "minimum": -0.04,
            "maximum": -0.04,
            "events_used": 1,
        },
        "contributions": {"ZIM": {"test_conflict": -0.04}},
    }
    monkeypatch.setitem(sys.modules, core.__name__, core)

    result = router_module.stress_test(
        StressTestParameters(
            data=[PortfolioHolding(symbol="ZIM", weight=1)],
            chokepoint="bab_el_mandeb_strait",
            event_type="conflict",
        )
    )

    assert result.results[0].event_id == "test_conflict"
    assert result.results[0].portfolio_car == pytest.approx(-0.04)
    assert result.results[0].contribution == pytest.approx(-0.04)
    assert result.results[0].portfolio_raw_return == pytest.approx((1.001**21) - 1)
    assert result.results[0].events_used == 1


def test_optimizer_wrapper_uses_current_weights_and_returns_changes(monkeypatch):
    _mock_event_returns(monkeypatch)
    market = sys.modules["openbb_chokepoints.data.market"]
    monkeypatch.setattr(
        market,
        "get_covariance",
        lambda symbols, lookback_days: pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.01]], index=symbols, columns=symbols
        ),
        raising=False,
    )
    captured = {}
    core = types.ModuleType("openbb_chokepoints.core.optimize")

    def min_change(w0, reactions, limit, bounds):
        captured.update(w0=w0, reactions=reactions, limit=limit, bounds=bounds)
        return [0.8, 0.2]

    core.min_change = min_change
    core.min_variance = lambda *args: [0.8, 0.2]
    monkeypatch.setitem(sys.modules, core.__name__, core)

    result = router_module.optimize(
        OptimizeParameters(
            data=[
                PortfolioHolding(symbol="ZIM", weight=0.5),
                PortfolioHolding(symbol="SPY", weight=0.5),
            ],
            chokepoint="bab_el_mandeb_strait",
            event_type="conflict",
            max_scenario_loss=0.03,
        )
    )

    assert captured["limit"] == pytest.approx(0.03)
    assert captured["bounds"] == [(0.0, 1.0), (0.0, 1.0)]
    assert result.results[0].new_weight == pytest.approx(0.8)
    assert result.results[1].new_weight == pytest.approx(0.2)
    assert result.results[0].turnover == pytest.approx(0.3)
    assert result.results[0].tracking_error is not None


def test_market_covariance_uses_shared_trading_dates_and_records_source_date(
    monkeypatch,
):
    market = sys.modules["openbb_chokepoints.data.market"]
    returns = pd.DataFrame(
        {
            "ZIM": [(index - 15) * 0.001 for index in range(30)],
            "SPY": [((index % 5) - 2) * 0.002 for index in range(30)],
        },
        index=pd.date_range("2024-01-01", periods=30),
    )
    returns.loc[returns.index[5], "SPY"] = None
    monkeypatch.setattr(market, "get_returns", lambda *args: returns)

    covariance = market.get_covariance(["ZIM", "SPY"], lookback_days=25)

    assert covariance.shape == (2, 2)
    assert covariance.attrs["data_through"] == returns.index[-1].date()
    assert covariance.attrs["data_pulled_at"].tzinfo is not None
    assert covariance.loc["ZIM", "SPY"] == pytest.approx(covariance.loc["SPY", "ZIM"])
