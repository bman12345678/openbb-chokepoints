# openbb-chokepoints

Chokepoint Risk Extension for the [OpenBB Platform](https://github.com/OpenBB-finance/OpenBB).

The goal is to link maritime chokepoint disruptions (Suez, Bab el-Mandeb,
Panama, Hormuz, Malacca, ...) to market reactions. The extension is designed
to add an `obb.chokepoints` namespace to the OpenBB Python interface and a
`/api/v1/chokepoints` prefix to the OpenBB REST API. The current implementation
includes a market-model event study and product-facing wrappers for traffic,
event, exposure, firm-exposure, stress, and optimizer workflows. Provider and
analytics modules merge independently through their own branches and reviews.

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
Start a new Python process after building. The market-model event study is
available in the current checkout. The PortWatch, catalog, detection, and
exposure modules have been pushed on separate branches, but are not yet part
of this product branch. The traffic, event, exposure, and firm-exposure
commands depend on those modules; stress and optimizer also depend on Manan's
analytics core being pushed and merged.

For local development with charts, install the optional charting dependencies:

```bash
pip install -e ".[dev,charting]"
```

This extension targets OpenBB's V4 package APIs (`openbb-core` 2.x,
`openbb-imf` 3.x, and related 2.x packages). In V4, PortWatch volume data is
accessed as `obb.economy.shipping.chokepoint_volume(provider="imf")`; verify
provider fields against the installed version when implementing loaders.
`openbb-economy` supplies the shared shipping commands, while `openbb-imf`
supplies their IMF data provider.

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
├── router.py            thin OpenBB command wrappers (no analytics)
├── models.py            Pydantic output models
├── chokepoints_views.py optional OpenBB chart views
├── charts.py            chart construction helpers
├── data/                PortWatch, market data, events and caches
├── core/                pure pandas/NumPy/SciPy analytics; never imports openbb
└── mcp/                 MCP prompts and agent-facing workflow assets
tests/
├── unit/            synthetic data, no network (run in CI)
└── integration/         OpenBB/provider tests marked `integration`
notebooks/               reproducible case studies and validation notebooks
```

The router extension and charting views use separate Python entry-point
groups. This keeps charts optional while allowing the same command results to
be used from Python and the REST API.

## Charts and command integration

The optional charting extension includes views for the traffic monitor, event
catalog, exposure lookup, event study, portfolio stress test, and optimizer.
The chart helpers take command result rows and do not fetch data or recompute
analytics. The full chart inputs, assumptions, output fields, and handoff
contracts are described in
[docs/charting-and-command-integration.md](docs/charting-and-command-integration.md).

The event-study chart includes a pointwise 95% band when its result includes
the cumulative CAR variance by day. Variance is returned by the market-model
core, so the chart draws the band from the same method used for the CAR test.

## Runnable examples

Install the development and charting dependencies and build the OpenBB command
interface first:

```bash
pip install -e ".[dev,charting]"
openbb-build
```

Start a fresh Python process after `openbb-build`. Each chart preview uses
synthetic data and writes an HTML file under `examples/output/` by default;
pass `--output /path/to/chart.html` to choose another location.

```bash
python examples/show_monitor_chart.py
python examples/show_events_chart.py
python examples/show_exposure_chart.py
python examples/show_sample_event_study.py
python examples/show_stress_test_chart.py
python examples/show_optimize_chart.py
```

These previews call the chart functions directly, so they do not contact any
provider or require the PortWatch and portfolio analytics handoffs. They are
illustrative only. The event-study fixture includes synthetic cumulative CAR
variance so its preview displays the confidence band.

To inspect all typed output models, run:

```bash
python examples/show_output_models.py
```

Pass `--model` to inspect one model, for example:

```bash
python examples/show_output_models.py --model MonitorData
```

To run a real OpenBB command and print its output table, use the command runner,
for example:

```bash
python examples/run_openbb_command.py event_study --symbols ZIM,FRO --event-date 2023-12-15 --chart
python examples/run_openbb_command.py monitor --chokepoint bab_el_mandeb_strait --vessel-type container --chart
python examples/run_openbb_command.py events --chokepoint bab_el_mandeb_strait --source both --chart
python examples/run_openbb_command.py exposure --chokepoint suez_canal --chart
python examples/run_openbb_command.py firm_exposure --symbol ZIM
python examples/run_openbb_command.py stress_test --portfolio ZIM:0.5,SPY:0.5 --chokepoint bab_el_mandeb_strait --event-type conflict --chart
python examples/run_openbb_command.py optimize --portfolio ZIM:0.5,SPY:0.5 --chokepoint bab_el_mandeb_strait --event-type conflict --max-scenario-loss 0.03 --chart
```

The live command runner requires provider access. `monitor`, `events`,
`exposure`, `firm_exposure`, `stress_test`, and `optimize` also need their
matching team-owned data or analytics modules merged; the synthetic chart
previews work before those handoffs are available.

## Roadmap

- Fama-French three-factor model, cross-sectional tests (BMP, Kolari-Pynnönen),
  sign and rank tests, overlapping-event clustering, date-shift sensitivity
- Stress test, optimizer, hedging
- Event catalog, disruption monitor, robust z-score, exposure lookup,
  IMF PortWatch data loaders, caching
- Workspace app layout and end-to-end widget verification, MCP workflow, PyPI release

## Disclaimer

This extension does scenario analysis based on past events. It is not
investment advice. Market data comes from third-party providers (Yahoo
Finance via `openbb-yfinance`) and may be incomplete or wrong.

## License

MIT. See [LICENSE](LICENSE).
