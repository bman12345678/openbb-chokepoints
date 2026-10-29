"""Price and return loader for the event study.

Prices come from ``obb.equity.price.historical`` with the yfinance provider,
adjusted for splits and dividends (the provider default is splits only).
Returns are simple returns, R_t = P_t / P_{t-1} - 1, computed on each
symbol's own trading calendar. Prices are never forward-filled: a filled
price on a holiday would create a fake 0% return.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

PROVIDER = "yfinance"
ADJUSTMENT = "splits_and_dividends"

# Calendar days per trading day, padded by 60% for holidays and thinly traded
# listings: 250 trading days -> about 1.6 calendar years.
_CALENDAR_PER_TRADING_DAY = 365.25 / 252 * 1.6
_PAD_DAYS = 7


def history_range(
    event_date: date,
    est_window: tuple[int, int],
    evt_window: tuple[int, int],
    today: date | None = None,
) -> tuple[date, date]:
    """Calendar (start, end) dates that cover both windows around event_date."""
    before = max(-est_window[0], -evt_window[0], 0)
    after = max(evt_window[1], 0)
    start = event_date - timedelta(
        days=math.ceil(before * _CALENDAR_PER_TRADING_DAY) + _PAD_DAYS
    )
    end = event_date + timedelta(
        days=math.ceil(after * _CALENDAR_PER_TRADING_DAY) + _PAD_DAYS
    )
    return start, min(end, today or date.today())


def get_returns(symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
    """Simple daily returns, one column per symbol, NaN where a symbol did not trade."""
    return prices_to_returns(fetch_prices(symbols, start, end))


def fetch_prices(symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
    """Adjusted close prices, one column per symbol, indexed by date."""
    # Imported here: importing openbb at module load would run the package
    # build while this extension is itself being loaded.
    from openbb import obb

    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    res = obb.equity.price.historical(
        symbol=",".join(symbols),
        start_date=start,
        end_date=end,
        provider=PROVIDER,
        adjustment=ADJUSTMENT,
    )
    df = res.to_df()
    if "symbol" not in df.columns:
        df["symbol"] = symbols[0]
    return long_to_wide(df)


def long_to_wide(df: pd.DataFrame, value: str = "close") -> pd.DataFrame:
    """Pivot long (date index, symbol column) prices to wide, without filling."""
    df = df.reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    wide = df.pivot_table(
        index=date_col, columns="symbol", values=value, aggfunc="last"
    )
    wide.index = pd.DatetimeIndex(wide.index).tz_localize(None).normalize()
    wide.columns.name = None
    return wide.sort_index()


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns per column on that column's own non-missing dates.

    A date where a symbol has no price stays NaN in the output; the next
    return spans the gap.
    """
    cols = {}
    for symbol in prices.columns:
        p = prices[symbol].dropna()
        cols[symbol] = (p / p.shift(1) - 1).iloc[1:]
    return pd.DataFrame(cols).sort_index()
