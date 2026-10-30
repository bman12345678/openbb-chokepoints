# Contributing

## Dev loop

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,charting]"
openbb-build                       # after adding/renaming a command or changing its signature

ruff check .                       # lint
ruff format .                      # format
pytest                             # unit + non-network tests (network tests are skipped)
pytest tests/unit --cov=openbb_chokepoints/core --cov-report=term-missing
pytest -m integration              # network tests (Yahoo Finance); run locally before a PR
```

The charting extra is optional. Core commands and tests must continue to work
when `openbb-charting` is not installed. For a minimal environment, use
`pip install -e ".[dev]"` instead.

Rules of thumb:

- `openbb_chokepoints/core/` is pure NumPy/pandas/SciPy. It must not import
  anything from `openbb`. `tests/unit/test_core_is_pure.py` enforces this.
- `router.py` holds thin wrappers only: parse inputs, load data, call `core`,
  return an `OBBject`.
- Data loaders live in `openbb_chokepoints/data/` and are the only place that
  talks to OpenBB providers.
- Unit tests use synthetic data with fixed seeds and no network. Mark anything
  that needs the network with `@pytest.mark.integration`.
- Integration tests should use recorded HTTP responses where possible. Record
  short responses with `pytest --record=http`, review the generated cassette,
  and commit it with the test so CI can replay the request offline.
- Rerun `openbb-build` and restart Python whenever a command signature changes,
  otherwise the static `obb` package is stale.

## Branches and pull requests

- `main` is protected by convention: no direct pushes after the initial
  scaffold. Work on a branch named `feature/<short-name>`, `fix/<short-name>`
  or `docs/<short-name>`.
- Keep commits small and descriptive (imperative mood, e.g. "Add market-model
  OLS fit").
- Open a pull request against `main` with `gh pr create`. Describe what changed,
  how it was tested, and how to run it.
- CI (ruff check, ruff format --check, unit tests on Python 3.10 and 3.12) must
  be green. CI also runs `openbb-build` to catch extension registration and
  generated-interface failures early.
- At least one teammate reviews and approves before merging. Authors do not
  merge their own PRs.
- Never commit virtual environments, caches, credentials, or `.env` files.
