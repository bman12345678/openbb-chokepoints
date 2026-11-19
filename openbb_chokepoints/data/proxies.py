"""Loader for ``proxies.yaml`` (industry -> sector -> ticker mapping)."""

from __future__ import annotations

from pathlib import Path

import yaml

PROXIES_PATH = Path(__file__).with_name("proxies.yaml")


def load_proxies(path: str | Path | None = None) -> dict:
    """Return ``{"industry_to_sector", "sector_proxies", "chokepoint_proxies"}``.

    Keys of ``industry_to_sector`` are lower-cased and stripped.
    """
    doc = yaml.safe_load(Path(path or PROXIES_PATH).read_text(encoding="utf-8")) or {}
    out = {
        "industry_to_sector": {
            str(k).strip().lower(): v
            for k, v in (doc.get("industry_to_sector") or {}).items()
        },
        "sector_proxies": dict(doc.get("sector_proxies") or {}),
        "chokepoint_proxies": dict(doc.get("chokepoint_proxies") or {}),
    }
    unknown = {
        s for s in out["industry_to_sector"].values() if s not in out["sector_proxies"]
    }
    if unknown:
        raise ValueError(f"sectors without proxies: {sorted(unknown)}")
    return out
