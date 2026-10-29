"""Router registration tests, plus a network test of obb.chokepoints.event_study."""

from importlib.metadata import entry_points

import pytest


def test_entry_point_registered():
    eps = entry_points(group="openbb_core_extension")
    ep = next(e for e in eps if e.name == "chokepoints")
    assert ep.value == "openbb_chokepoints.router:router"


def test_router_loads():
    from openbb_core.app.router import Router

    from openbb_chokepoints.router import router

    assert isinstance(router, Router)
    assert router.description == "Chokepoint disruption risk analytics."


def test_event_study_route_registered():
    from openbb_chokepoints.router import router

    paths = {route.path: route for route in router.api_router.routes}
    route = paths["/event_study"]
    assert route.methods == {"GET"}
    first_example = route.openapi_extra["examples"][0]
    assert set(first_example.parameters) == {"symbols", "event_date"}


@pytest.mark.integration
def test_event_study_zim_fro():
    """Live Yahoo Finance data; checks the schema only, never the numbers."""
    from openbb import obb

    from openbb_chokepoints.models import EventStudyData

    res = obb.chokepoints.event_study(symbols="ZIM,FRO", event_date="2023-12-15")

    assert res.provider == "yfinance"
    assert any("not investment advice" in w.message for w in res.warnings)
    assert {r.symbol for r in res.results} == {"ZIM", "FRO"}
    for row in res.results:
        assert isinstance(row, EventStudyData)
        assert row.window in {"(-5, 20)", "(-1, 1)", "(0, 5)", "(0, 20)"}
        assert row.day0.isoformat() >= "2023-12-15"
        assert row.n_est >= 120
        assert len(row.ar_by_day) == 26
        assert row.p_value is None or 0.0 <= row.p_value <= 1.0

    df = res.to_df()
    assert set(EventStudyData.model_fields) <= set(df.columns) | {df.index.name}
