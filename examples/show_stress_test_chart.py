"""Render the historical portfolio stress chart using synthetic event rows.

Run: python examples/show_stress_test_chart.py [--output /tmp/stress-test.html]
"""

from chart_demo_utils import save_chart

from openbb_chokepoints.charts import stress_test_chart


def main() -> None:
    rows = [
        {
            "event_id": event_id,
            "symbol": symbol,
            "contribution": contribution,
            "portfolio_car": portfolio_car,
            "scenario_median": -0.025,
            "events_used": 3,
        }
        for event_id, portfolio_car, holdings in [
            ("red-sea-demo", -0.04, {"ZIM": -0.03, "FRO": -0.015, "SPY": 0.005}),
            ("suez-demo", -0.01, {"ZIM": -0.01, "FRO": 0.004, "SPY": -0.004}),
            ("panama-demo", -0.025, {"ZIM": -0.02, "FRO": -0.01, "SPY": 0.005}),
        ]
        for symbol, contribution in holdings.items()
    ]
    save_chart(stress_test_chart(rows), "stress-test.html")


if __name__ == "__main__":
    main()
