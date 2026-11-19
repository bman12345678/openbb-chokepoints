"""Check that every ticker in proxies.yaml has usable price history.

Needs ``pip install yfinance``.

    python scripts/check_proxy_coverage.py [--start 2019-01-01]
"""

from __future__ import annotations

import argparse
import sys

from openbb_chokepoints.data.proxies import load_proxies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--min-rows", type=int, default=500)
    args = ap.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed: pip install yfinance")
        return 2

    px = load_proxies()
    tickers = sorted(
        {t for v in px["sector_proxies"].values() for t in v}
        | {t for v in px["chokepoint_proxies"].values() for t in v}
    )
    bad = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=args.start, auto_adjust=True)
            n = len(hist)
            first = hist.index.min().date() if n else None
        except Exception as exc:  # network or symbol errors
            n, first = 0, f"error: {exc}"
        status = "ok" if n >= args.min_rows else "THIN"
        if status != "ok":
            bad.append(t)
        print(f"{t:14s} {n:6d} rows  first={first}  {status}")
    print(f"\n{len(bad)} of {len(tickers)} tickers thin/missing: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
