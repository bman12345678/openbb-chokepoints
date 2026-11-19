"""Render the portfolio optimizer chart using synthetic result rows.

Run: python examples/show_optimize_chart.py [--output /tmp/optimizer.html]
"""

from chart_demo_utils import save_chart

from openbb_chokepoints.charts import optimize_chart


def main() -> None:
    rows = [
        {
            "symbol": symbol,
            "current_weight": current,
            "new_weight": proposed,
            "loss_before": -0.061,
            "loss_after": -0.029,
            "turnover": 0.18,
        }
        for symbol, current, proposed in [
            ("ZIM", 0.40, 0.22),
            ("FRO", 0.25, 0.18),
            ("SPY", 0.35, 0.60),
        ]
    ]
    save_chart(optimize_chart(rows), "optimizer.html")


if __name__ == "__main__":
    main()
