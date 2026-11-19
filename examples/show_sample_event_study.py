"""Render the synthetic event-study result fixture as a local HTML chart.

Run from the repository root after installing the optional charting extra:

    python examples/show_sample_event_study.py
    python examples/show_sample_event_study.py --output /tmp/event-study.html
"""

from __future__ import annotations

import json
from pathlib import Path

from chart_demo_utils import save_chart

from openbb_chokepoints.chokepoints_views import ChokepointsViews


def main() -> None:
    """Render the fixture through the chart view into a standalone HTML file."""
    # The fixture mirrors EventStudyData rows and the command's event_window input.
    fixture_path = Path(__file__).with_name("event_study_sample.json")
    sample = json.loads(fixture_path.read_text(encoding="utf-8"))
    figure, _ = ChokepointsViews.chokepoints_event_study(
        obbject_item=sample["results"],
        event_window=sample["event_window"],
        sample_mode=True,
    )
    print(sample["sample_notice"])
    save_chart(figure, "event-study.html")


if __name__ == "__main__":
    main()
