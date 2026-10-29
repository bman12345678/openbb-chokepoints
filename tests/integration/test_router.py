"""Router registration tests."""

from importlib.metadata import entry_points


def test_entry_point_registered():
    eps = entry_points(group="openbb_core_extension")
    ep = next(e for e in eps if e.name == "chokepoints")
    assert ep.value == "openbb_chokepoints.router:router"


def test_router_loads():
    from openbb_core.app.router import Router

    from openbb_chokepoints.router import router

    assert isinstance(router, Router)
    assert router.description == "Chokepoint disruption risk analytics."
