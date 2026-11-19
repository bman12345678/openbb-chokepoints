"""Run the 18-point detection grid against the event catalog.

Writes ``validation_grid.csv`` and ``validation_summary.md`` to ``--out``.

    python scripts/run_detection_validation.py [--out reports] [--refresh]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from openbb_chokepoints.core.validation import run_grid, select_params
from openbb_chokepoints.data.events import load_events
from openbb_chokepoints.data.portwatch import get_chokepoint_volume

METRIC_COLS = [
    "n_events",
    "hits",
    "recall",
    "n_episodes",
    "false_alarms",
    "precision",
    "f1",
    "median_delay_days",
    "fa_per_year",
]
PARAM_COLS = ["smooth_days", "tau", "k", "n"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = load_events()
    scope = set(events["chokepoint"])
    for rel in events["related_chokepoints"]:
        scope.update(rel)
    volume = get_chokepoint_volume(sorted(scope), refresh=args.refresh)

    results = run_grid(volume, events)
    results.to_csv(out / "validation_grid.csv", index=False)
    best = select_params(results)
    mask = pd.Series(True, index=results.index)
    for c in PARAM_COLS:
        mask &= results[c] == best[c]
    chosen = results[mask].set_index("period")[METRIC_COLS]

    lines = [
        "# Detection validation",
        "",
        f"Data through {volume.attrs.get('data_through')}, "
        f"pulled {volume.attrs.get('data_pulled_at')}.",
        "",
        "Selected on 2019-2022 by max F1 (ties: fewer false alarms, larger tau, "
        "larger n):",
        "",
        "`" + ", ".join(f"{c}={best[c]}" for c in PARAM_COLS) + "`",
        "",
        "```",
        chosen.round(3).to_string(),
        "```",
    ]
    (out / "validation_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
