"""Shared helpers for standalone, synthetic chart previews."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def save_chart(figure: Any, filename: str) -> None:
    """Write a self-contained chart preview to examples/output by default."""
    parser = argparse.ArgumentParser(description="Save this synthetic chart preview.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("output") / filename,
        help="HTML destination (defaults to examples/output/<chart>.html).",
    )
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        figure.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8"
    )
    print("Synthetic demonstration only; values are not observed market data.")
    print(f"Wrote chart to {output.resolve()}")
