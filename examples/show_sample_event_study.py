"""Render the synthetic event-study result fixture as a local HTML chart.

Run from the repository root after installing the optional charting extra:

    python examples/show_sample_event_study.py
    python examples/show_sample_event_study.py --output /tmp/event-study.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openbb_chokepoints.chokepoints_views import ChokepointsViews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_event_study.html"),
        help="HTML output path (default: ./sample_event_study.html)",
    )
    args = parser.parse_args()

    fixture_path = Path(__file__).with_name("event_study_sample.json")
    sample = json.loads(fixture_path.read_text(encoding="utf-8"))
    figure, _ = ChokepointsViews.chokepoints_event_study(
        obbject_item=sample["results"],
        event_window=sample["event_window"],
        sample_mode=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        figure.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8"
    )
    print(sample["sample_notice"])
    print(f"Wrote chart to {args.output.resolve()}")


if __name__ == "__main__":
    main()
