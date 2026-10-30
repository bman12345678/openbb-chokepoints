"""Chart adapter tests using synthetic feature-branch result data."""

import json
from pathlib import Path

import pytest


def test_event_study_chart_uses_full_window_rows_and_marks_day_zero():
    """Check the chart behavior promised by the event-study chart work package."""
    pytest.importorskip("openbb_charting")

    from openbb_chokepoints.charts import event_study_chart

    sample_path = Path(__file__).parents[2] / "examples" / "event_study_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = sample["results"]
    for row in rows:
        assert len(row["ar_by_day"]) == 26
        assert sum(row["ar_by_day"]) == pytest.approx(row["car"])

    # The live command returns subwindow rows too. A chart should retain one
    # full-window trace per symbol rather than drawing each symbol repeatedly.
    rows += [
        {
            **row,
            "window": "(0, 5)",
            "car": sum(row["ar_by_day"][5:11]),
        }
        for row in sample["results"]
    ]
    figure = event_study_chart(
        rows,
        tuple(sample["event_window"]),
        sample_mode=True,
    )

    assert {trace.name for trace in figure.data} == {"FRO", "ZIM"}
    assert all(len(trace.x) == 26 for trace in figure.data)
    assert all(
        trace.y[5] == pytest.approx(sum(trace.customdata[:6]) * 100)
        for trace in figure.data
    )
    assert any(shape.x0 == 0 and shape.x1 == 0 for shape in figure.layout.shapes)
    assert "synthetic sample" in figure.layout.title.text
