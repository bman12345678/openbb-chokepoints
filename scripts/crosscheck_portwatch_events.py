"""Compare our catalog with PortWatch's own disruption database.

For each catalog disruption, print PortWatch events whose start date is within
``--tolerance`` days of ``news_date`` so dates can be sanity-checked by hand.
(PortWatch event types/countries differ from ours; this is a hint, not truth.)

    python scripts/crosscheck_portwatch_events.py [--tolerance 10]
"""

from __future__ import annotations

import argparse

import pandas as pd

from openbb_chokepoints.data.events import load_events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=int, default=10)
    args = ap.parse_args()

    from openbb import obb  # imported late: scripts only

    pw = obb.economy.shipping.disruption_events(provider="imf").to_df()
    pw["from"] = pd.to_datetime(
        pw["fromdate"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    cat = load_events(event_type="disruption")
    for _, e in cat.iterrows():
        gap = (pw["from"] - e["news_date"]).abs().dt.days
        near = pw[gap <= args.tolerance]
        print(f"\n{e['id']}  news_date={e['news_date'].date()}")
        if near.empty:
            print("  no PortWatch event within tolerance")
        for _, r in near.iterrows():
            print(f"  {r['from'].date()}  {r['eventtype']}  {r['eventname']}")


if __name__ == "__main__":
    main()
