"""Render the exposure chart using synthetic PortWatch-shaped rows.

Run: python examples/show_exposure_chart.py [--output /tmp/exposure.html]
"""

from chart_demo_utils import save_chart

from openbb_chokepoints.charts import exposure_chart


def main() -> None:
    rows = [
        {
            "chokepoint": "suez_canal",
            "category": "vessel_type",
            "name": name,
            "share_count": share,
            "share_capacity": capacity,
            "rank": None,
            "proxy_symbols": [],
        }
        for name, share, capacity in [
            ("tanker", 0.24, 0.42),
            ("container", 0.31, 0.28),
            ("dry_bulk", 0.22, 0.18),
            ("general_cargo", 0.13, 0.08),
            ("roro", 0.10, 0.04),
        ]
    ]
    rows.extend(
        {
            "chokepoint": "suez_canal",
            "category": "industry",
            "name": name,
            "share_count": None,
            "share_capacity": None,
            "rank": rank,
            "proxy_symbols": [],
        }
        for rank, name in enumerate(("Manufacturing", "Energy", "Transport"), 1)
    )
    save_chart(exposure_chart(rows), "exposure.html")


if __name__ == "__main__":
    main()
