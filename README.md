# openbb-chokepoints

Chokepoint Risk Extension for the [OpenBB Platform](https://github.com/OpenBB-finance/OpenBB).

The goal is to link maritime chokepoint disruptions (Suez, Bab el-Mandeb,
Panama, Hormuz, Malacca, ...) to market reactions. The extension is designed
to add an `obb.chokepoints` namespace to the OpenBB Python interface and a
`/api/v1/chokepoints` prefix to the OpenBB REST API. The current `main` branch
is a scaffold; the namespace gains commands as features are merged.

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
Start a new Python process after building. On `main`, no commands have been
merged yet, so there is not a usable `obb.chokepoints` command to call.

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

No commands have been merged yet.

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

## Sample event-study chart

The event-study branch returns rows with `symbol`, `window`, and
`ar_by_day` fields. A synthetic ZIM/FRO fixture using that shape is provided
for UI work; the values are illustrative and are not market observations.
After installing the optional charting dependencies, render the demo with:

```bash
python examples/show_sample_event_study.py
```

This writes `sample_event_study.html` in the current directory. Pass
`--output /path/to/file.html` to choose another location.

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
