"""List the industry labels PortWatch actually uses and which ones
proxies.yaml does not map yet.

    python scripts/list_industries.py
"""

from __future__ import annotations

from openbb_chokepoints.core.exposure import industry_ranks, unmapped_industries
from openbb_chokepoints.data.portwatch import get_chokepoint_info
from openbb_chokepoints.data.proxies import load_proxies


def main() -> None:
    info = get_chokepoint_info()
    proxies = load_proxies()
    ranks = industry_ranks(info, proxies["industry_to_sector"])
    counts = ranks.groupby("industry").size().sort_values(ascending=False)
    print("Industry labels in PortWatch (count of chokepoint-ranks):")
    print(counts.to_string())
    missing = unmapped_industries(ranks)
    print(f"\nUnmapped ({len(missing)}): add these to industry_to_sector:")
    for label in missing:
        print(f'  "{label.lower()}": <sector>')


if __name__ == "__main__":
    main()
