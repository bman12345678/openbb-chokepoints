"""Render the event timeline using synthetic catalog rows.

Run: python examples/show_events_chart.py [--output /tmp/events.html]
"""

from chart_demo_utils import save_chart

from openbb_chokepoints.charts import events_chart


def main() -> None:
    rows = [
        {
            "event_id": "red-sea-conflict-demo",
            "news_date": "2023-12-15",
            "chokepoint": "bab_el_mandeb_strait",
            "event_type": "conflict",
            "direction": "disruption",
            "confidence": "high",
            "description": "Synthetic example: disruption begins",
        },
        {
            "event_id": "suez-drought-demo",
            "news_date": "2024-01-12",
            "chokepoint": "suez_canal",
            "event_type": "climate",
            "direction": "disruption",
            "confidence": "medium",
            "description": "Synthetic example: traffic pressure increases",
        },
        {
            "event_id": "red-sea-recovery-demo",
            "news_date": "2024-03-01",
            "chokepoint": "bab_el_mandeb_strait",
            "event_type": "resolution",
            "direction": "recovery",
            "confidence": "medium",
            "description": "Synthetic example: recovery marker is hollow",
        },
    ]
    save_chart(events_chart(rows), "events.html")


if __name__ == "__main__":
    main()
