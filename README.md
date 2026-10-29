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

No commands have been merged yet.

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
