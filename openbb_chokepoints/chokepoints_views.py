"""Optional chart views for commands in :mod:`openbb_chokepoints.router`.

OpenBB charting discovers public static methods named after command routes.
Add a method when the corresponding command is implemented; keeping this
module free of charting imports allows the base package to work without the
optional ``charting`` extra.
"""


class ChokepointsViews:
    """OpenBB chart hooks for routes, as specified by the charting plan."""

    @staticmethod
    def chokepoints_event_study(**kwargs):
        """Adapt event-study result rows to OpenBB's figure/content pair."""
        import json

        from openbb_chokepoints.charts import event_study_chart

        event_window = kwargs.get("event_window", (-5, 20))
        if not isinstance(event_window, (tuple, list)) or len(event_window) != 2:
            event_window = (-5, 20)

        figure = event_study_chart(
            kwargs.get("obbject_item") or [],
            (int(event_window[0]), int(event_window[1])),
            sample_mode=bool(kwargs.get("sample_mode", False)),
            theme=kwargs.get("theme", "dark"),
        )
        content = json.loads(figure.show(external=True).to_json())
        content["config"] = {"displayModeBar": False}
        return figure, content
