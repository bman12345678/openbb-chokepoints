"""Render the traffic monitor using synthetic observations.

Run: python examples/show_monitor_chart.py [--output /tmp/monitor.html]
"""

from datetime import date, timedelta

from chart_demo_utils import save_chart

from openbb_chokepoints.charts import monitor_chart


def main() -> None:
    today = date(2026, 11, 18)
    counts = [98, 101, 103, 100, 97, 99, 102, 96, 94, 91, 72, 70, 74, 69]
    rows = [
        {
            "chokepoint": "bab_el_mandeb_strait",
            "vessel_type": "container",
            "as_of": today - timedelta(days=len(counts) - index - 1),
            "avg_7d": count,
            "baseline_median": 100,
            "normal_low": 82,
            "normal_high": 118,
            "robust_z": -3.1 if count < 80 else 0.1,
            "flagged": count < 80,
            "data_pulled_at": "2026-11-18T15:00:00Z",
            "data_through": today,
        }
        for index, count in enumerate(counts)
    ]
    save_chart(monitor_chart(rows), "monitor.html")


if __name__ == "__main__":
    main()
