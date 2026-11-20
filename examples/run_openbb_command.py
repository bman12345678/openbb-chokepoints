"""Run one real OpenBB command and print its typed result table.

Build the OpenBB interface first with ``openbb-build``. Example:

    python examples/run_openbb_command.py event_study \\
        --symbols ZIM,FRO --event-date 2023-12-15 --chart
    python examples/run_openbb_command.py monitor \\
        --chokepoint bab_el_mandeb_strait --vessel-type container
    python examples/run_openbb_command.py stress_test \\
        --portfolio ZIM:0.5,SPY:0.5 \\
        --chokepoint bab_el_mandeb_strait --event-type conflict

The monitor, event, exposure, stress, and optimizer calls need the matching
teammate-owned provider or analytics modules to be merged. The event-study
route can run from this branch when its OpenBB market-data provider is installed.
"""

from __future__ import annotations

import argparse


def _portfolio(value: str) -> list[dict[str, float | str]]:
    holdings = []
    for item in value.split(","):
        symbol, separator, weight = item.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(
                "Portfolio must use SYMBOL:WEIGHT pairs, e.g. ZIM:0.5,SPY:0.5."
            )
        try:
            parsed_weight = float(weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid weight in {item!r}.") from exc
        holdings.append({"symbol": symbol.strip().upper(), "weight": parsed_weight})
    if not holdings:
        raise argparse.ArgumentTypeError("Provide at least one portfolio holding.")
    return holdings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    monitor = commands.add_parser("monitor", help="Traffic monitor result rows")
    monitor.add_argument("--chokepoint")
    monitor.add_argument("--vessel-type")
    monitor.add_argument("--lookback-days", type=int, default=90)
    monitor.add_argument("--threshold", type=float, default=3.0)
    monitor.add_argument("--chart", action="store_true")

    events = commands.add_parser("events", help="Catalog and detected events")
    events.add_argument("--chokepoint")
    events.add_argument("--event-type")
    events.add_argument("--direction", choices=("disruption", "recovery"))
    events.add_argument(
        "--source", choices=("catalog", "detected", "both"), default="both"
    )
    events.add_argument("--chart", action="store_true")

    exposure = commands.add_parser("exposure", help="Vessel and industry exposure")
    exposure.add_argument("--chokepoint", required=True)
    exposure.add_argument(
        "--measure", choices=("share_count", "share_capacity"), default="share_count"
    )
    exposure.add_argument("--chart", action="store_true")

    firm_exposure = commands.add_parser(
        "firm_exposure", help="Find mapped route and sector links for a symbol"
    )
    firm_exposure.add_argument("--symbol", required=True)
    firm_exposure.add_argument("--chokepoint")

    study = commands.add_parser("event_study", help="Market-model event study")
    study.add_argument("--symbols", default="ZIM,FRO")
    study.add_argument("--event-date", required=True)
    study.add_argument("--market-symbol", default="SPY")
    study.add_argument("--chart", action="store_true")

    stress = commands.add_parser("stress_test", help="Historical portfolio stress")
    stress.add_argument("--portfolio", type=_portfolio, required=True)
    stress.add_argument("--chokepoint", required=True)
    stress.add_argument("--event-type", required=True)
    stress.add_argument(
        "--direction", choices=("disruption", "recovery"), default="disruption"
    )
    stress.add_argument("--chart", action="store_true")

    optimize = commands.add_parser("optimize", help="Scenario-constrained weights")
    optimize.add_argument("--portfolio", type=_portfolio, required=True)
    optimize.add_argument("--chokepoint", required=True)
    optimize.add_argument("--event-type", required=True)
    optimize.add_argument("--max-scenario-loss", type=float, default=0.03)
    optimize.add_argument(
        "--objective", choices=("min_change", "min_variance"), default="min_change"
    )
    optimize.add_argument(
        "--constraint", choices=("average", "every_event"), default="average"
    )
    optimize.add_argument(
        "--direction", choices=("disruption", "recovery"), default="disruption"
    )
    optimize.add_argument("--chart", action="store_true")
    return parser


def main() -> None:
    args = vars(_parser().parse_args())
    command_name = args.pop("command")
    chart = args.pop("chart", False)
    if command_name in {"stress_test", "optimize"}:
        portfolio = args.pop("portfolio")
        args["data"] = portfolio

    try:
        from openbb import obb

        command = getattr(obb.chokepoints, command_name)
        result = command(**args, chart=chart)
    except ImportError as exc:
        raise SystemExit(
            "OpenBB is not installed in this environment. Install the extension, "
            "then run openbb-build and start a new Python process."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"{command_name} could not run: {exc}") from exc

    frame = result.to_df()
    print(frame.to_string(index=False) if not frame.empty else "No rows returned.")


if __name__ == "__main__":
    main()
