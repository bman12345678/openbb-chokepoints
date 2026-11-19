"""Print a synthetic example row and field schema for each output model.

Examples:
    python examples/show_output_models.py
    python examples/show_output_models.py --model MonitorData
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from openbb_chokepoints.models import (
    EventCatalogData,
    EventStudyData,
    ExposureData,
    MonitorData,
    OptimizeData,
    StressTestData,
)

SAMPLES = {
    "EventStudyData": (
        EventStudyData,
        {
            "symbol": "ZIM",
            "event_date": "2023-12-15",
            "day0": "2023-12-15",
            "window": "(-5, 20)",
            "car": -0.042,
            "car_t": -1.4,
            "p_value": 0.16,
            "alpha": 0.0002,
            "beta": 1.2,
            "sigma_eps": 0.015,
            "n_est": 221,
            "ar_by_day": [-0.004, 0.003, -0.02],
            "car_var_by_day": [0.0001, 0.0002, 0.0003],
        },
    ),
    "MonitorData": (
        MonitorData,
        {
            "chokepoint": "bab_el_mandeb_strait",
            "vessel_type": "container",
            "as_of": "2026-11-18",
            "avg_7d": 72.0,
            "baseline_median": 100.0,
            "normal_low": 82.0,
            "normal_high": 118.0,
            "robust_z": -3.1,
            "pct_vs_baseline": -0.28,
            "flagged": True,
        },
    ),
    "EventCatalogData": (
        EventCatalogData,
        {
            "event_id": "red-sea-demo",
            "chokepoint": "bab_el_mandeb_strait",
            "event_type": "conflict",
            "direction": "disruption",
            "news_date": "2023-12-15",
            "confidence": "high",
            "description": "Synthetic catalog example",
            "sources": ["https://example.invalid/source"],
        },
    ),
    "ExposureData": (
        ExposureData,
        {
            "chokepoint": "suez_canal",
            "category": "vessel_type",
            "name": "tanker",
            "share_count": 0.24,
            "share_capacity": 0.42,
            "proxy_symbols": ["XLE"],
        },
    ),
    "StressTestData": (
        StressTestData,
        {
            "event_id": "red-sea-demo",
            "symbol": "ZIM",
            "weight": 0.5,
            "contribution": -0.021,
            "portfolio_car": -0.035,
            "scenario_median": -0.025,
            "scenario_min": -0.06,
            "scenario_max": 0.01,
            "events_used": 3,
        },
    ),
    "OptimizeData": (
        OptimizeData,
        {
            "symbol": "ZIM",
            "current_weight": 0.5,
            "new_weight": 0.3,
            "change": -0.2,
            "loss_before": -0.06,
            "loss_after": -0.03,
            "turnover": 0.2,
            "tracking_error": 0.08,
        },
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *SAMPLES), default="all")
    args = parser.parse_args()
    names = SAMPLES if args.model == "all" else [args.model]
    for name in names:
        model, sample = SAMPLES[name]
        result = model.model_validate(sample)
        print(f"\n## {name}: sample output row")
        print(pd.DataFrame([result.model_dump(mode="json")]).to_string(index=False))
        print("Field schema:")
        print(json.dumps(model.model_json_schema(), indent=2, default=str))


if __name__ == "__main__":
    main()
