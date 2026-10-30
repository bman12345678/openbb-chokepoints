"""Product/platform chart-wiring tests using the event-study result shape.

These tests use the event-study result shape from the feature branch. They do
not exercise the event-study calculations or fetch provider data.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SAMPLE_PATH = ROOT / "examples" / "event_study_sample.json"


def test_sample_rows_match_event_study_branch_shape():
    """Keep the sample fixture compatible with event-study result rows."""
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    required = {
        "symbol",
        "event_date",
        "day0",
        "window",
        "car",
        "car_t",
        "p_value",
        "alpha",
        "beta",
        "sigma_eps",
        "n_est",
        "ar_by_day",
    }

    assert sample["event_window"] == [-5, 20]
    assert "Synthetic" in sample["sample_notice"]
    assert len(sample["results"]) == 2
    for row in sample["results"]:
        assert required <= row.keys()
        assert row["window"] == "(-5, 20)"
        assert len(row["ar_by_day"]) == 26
        assert sum(row["ar_by_day"]) == pytest.approx(row["car"])


def test_chart_view_wraps_figure_as_openbb_content(monkeypatch):
    """Check the route-named view returns OpenBB's chart adapter contract."""
    pytest.importorskip("openbb_charting")

    from openbb_chokepoints import charts
    from openbb_chokepoints.chokepoints_views import ChokepointsViews

    class FakeFigure:
        """Stand-in for OpenBBFigure while testing view serialization."""

        def show(self, *, external):
            """Match the external serialization call used by OpenBB chart views."""
            assert external is True
            return self

        def to_json(self):
            """Return the serialized figure payload expected by the adapter."""
            return '{"data": [], "layout": {}}'

    figure = FakeFigure()
    received = {}

    def fake_chart(rows, event_window, *, sample_mode, theme):
        """Capture the chart inputs forwarded by the OpenBB view."""
        received.update(
            rows=rows,
            event_window=event_window,
            sample_mode=sample_mode,
            theme=theme,
        )
        return figure

    monkeypatch.setattr(charts, "event_study_chart", fake_chart)
    rows = [{"symbol": "ZIM", "ar_by_day": [0.01]}]

    returned_figure, content = ChokepointsViews.chokepoints_event_study(
        obbject_item=rows,
        event_window=[-1, 0],
        sample_mode=True,
        theme="light",
    )

    assert returned_figure is figure
    assert content == {"data": [], "layout": {}, "config": {"displayModeBar": False}}
    assert received == {
        "rows": rows,
        "event_window": (-1, 0),
        "sample_mode": True,
        "theme": "light",
    }


def test_sample_renderer_writes_standalone_html(tmp_path):
    """Verify the sample workflow produces a labeled, local HTML chart."""
    pytest.importorskip("openbb_charting")
    output = tmp_path / "nested" / "event-study.html"
    script = ROOT / "examples" / "show_sample_event_study.py"

    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    html = output.read_text(encoding="utf-8")
    assert "Synthetic demonstration data only" in result.stdout
    assert "Cumulative Abnormal Returns" in html
    assert "Illustrative synthetic data" in html
    assert "plotly" in html.lower()


def test_chart_modules_import_without_optional_charting_dependency():
    """Protect the plan's requirement that charting remains an optional extra."""
    # Simulate a minimal install even when this test environment has the extra.
    code = '''
import importlib.abc
import sys

class BlockCharting(importlib.abc.MetaPathFinder):
    """Pretend the optional OpenBB charting package is not installed."""

    def find_spec(self, fullname, path=None, target=None):
        """Block charting imports while leaving the base extension importable."""
        if fullname == 'openbb_charting' or fullname.startswith('openbb_charting.'):
            raise ModuleNotFoundError('blocked optional charting dependency')
        return None

sys.meta_path.insert(0, BlockCharting())
import openbb_chokepoints.charts
import openbb_chokepoints.chokepoints_views
assert not any(name.startswith('openbb_charting') for name in sys.modules)
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_charting_entrypoint_names_the_event_study_view():
    """Check OpenBB can discover the route-named chart view from its entry point."""
    pytest.importorskip("openbb_charting")

    from importlib.metadata import entry_points

    from openbb_chokepoints.chokepoints_views import ChokepointsViews

    charting = next(
        ep
        for ep in entry_points(group="openbb_charting_extension")
        if ep.name == "chokepoints"
    )
    assert charting.load() is ChokepointsViews
    assert callable(charting.load().chokepoints_event_study)
