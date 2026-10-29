# openbb-chokepoints

Chokepoint Risk Extension for the [OpenBB Platform](https://github.com/OpenBB-finance/OpenBB).

The goal is to link maritime chokepoint disruptions (Suez, Bab el-Mandeb,
Panama, Hormuz, Malacca, ...) to market reactions. The extension adds an
`obb.chokepoints` namespace to the OpenBB Python interface and a
`/api/v1/chokepoints` prefix to the OpenBB REST API.

| | |
|---|---|
| Package | `openbb-chokepoints` |
| Import | `openbb_chokepoints` |
| Python namespace | `obb.chokepoints` |
| REST prefix | `/api/v1/chokepoints` |

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/bman12345678/openbb-chokepoints.git
cd openbb-chokepoints
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
openbb-build                       # regenerates the static `obb` package
```

`openbb-build` is required after installing or removing any OpenBB extension.
Without it, `obb.chokepoints` does not appear in the Python interface. Start a
new Python process after building.

```python
from openbb import obb

obb.chokepoints  # lists the available commands
```

The REST API is served by `openbb-api` (from `openbb-platform-api`) at
`http://127.0.0.1:6900/api/v1/chokepoints/...`.

## Commands

### `obb.chokepoints.event_study`

Market-model event study (MacKinlay 1997; Campbell, Lo & MacKinlay 1997,
ch. 4). For each symbol it fits `R_i = alpha + beta * R_m` by OLS over the
estimation window, then measures abnormal returns `AR = R_i - (alpha + beta * R_m)`
around a date you supply. It reports cumulative abnormal returns (CAR) with
t-tests that include the estimation error in alpha and beta.

| Parameter | Default | Meaning |
|---|---|---|
| `symbols` | required | List of tickers, or a comma-separated string |
| `event_date` | required | Day 0 is each symbol's first trading day on or after this date |
| `market_symbol` | `"SPY"` | Market index. For non-US stocks use the local index (e.g. `^OMXC25`, `^N225`) |
| `estimation_window` | `(-250, -30)` | Trading days relative to day 0 used to fit the model |
| `event_window` | `(-5, 20)` | Trading days relative to day 0 for abnormal returns |
| `min_estimation_obs` | `120` | Symbols with fewer estimation days are skipped with a warning |

Output: one row per symbol and CAR window (the full event window, plus
`(-1, 1)`, `(0, 5)` and `(0, 20)` when they fit inside it), with `car`,
`car_t`, `p_value`, `alpha`, `beta`, `sigma_eps`, `n_est`, `day0` and the
daily `ar_by_day`. Returns are decimals (0.05 = 5%). Prices come from Yahoo
Finance, adjusted for splits and dividends, and are never forward-filled.

```python
from openbb import obb

res = obb.chokepoints.event_study(symbols=["ZIM", "FRO"], event_date="2023-12-15")
res.to_df()[["symbol", "window", "car", "car_t", "p_value"]]
```

REST: `GET /api/v1/chokepoints/event_study?symbols=ZIM,FRO&event_date=2023-12-15`.
Pass a window as repeated parameters: `&event_window=-2&event_window=10`.

## Project layout

```
openbb_chokepoints/
├── router.py        thin OpenBB command wrappers (no math)
├── models.py        Pydantic output models
├── data/            data loaders (OpenBB providers)
└── core/            pure NumPy/pandas analytics; never imports openbb
tests/
├── unit/            synthetic data, no network (run in CI)
└── integration/     OpenBB/network tests (network ones marked `integration`)
```

## Roadmap

- Fama-French three-factor model, cross-sectional tests (BMP, Kolari-Pynnönen),
  sign and rank tests, overlapping-event clustering, date-shift sensitivity
- Stress test, optimizer, hedging
- Event catalog, disruption monitor, robust z-score, exposure lookup,
  IMF PortWatch data loaders, caching
- Charts, OpenBB Workspace widgets, MCP configuration, PyPI release

## Disclaimer

This extension does scenario analysis based on past events. It is not
investment advice. Market data comes from third-party providers (Yahoo
Finance via `openbb-yfinance`) and may be incomplete or wrong.

## License

MIT. See [LICENSE](LICENSE).
