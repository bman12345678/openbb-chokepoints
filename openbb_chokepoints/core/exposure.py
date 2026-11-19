"""Exposure lookup and proxy mapping (WP4). Pure pandas: no OpenBB imports.

What PortWatch gives us (``get_chokepoint_info``) is, per chokepoint, vessel
counts by type and the *top three industries by rank* (not shares of traffic).
So this module reports:

* ``vessel_type_shares``: real shares of vessel counts by type;
* ``sector_exposure``: an ordinal score from industry rank (rank 1 > 2 > 3),
  explicitly NOT a share of traffic;
* proxies (tickers) for a chokepoint, from route-specific names plus the
  sectors that score highest there.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

TYPE_COLUMNS = {
    "tanker": "vessel_count_tanker",
    "container": "vessel_count_container",
    "general_cargo": "vessel_count_general_cargo",
    "dry_bulk": "vessel_count_dry_bulk",
    "roro": "vessel_count_roro",
}
RANK_COLUMNS = ("industry_top1", "industry_top2", "industry_top3")
DEFAULT_RANK_WEIGHTS = (3.0, 2.0, 1.0)


def _need(info: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in info.columns]
    if missing:
        raise ValueError(f"info table is missing columns: {missing}")


def vessel_type_shares(info: pd.DataFrame, *, tolerance: float = 0.05) -> pd.DataFrame:
    """Share of vessel counts by type, one row per chokepoint.

    Shares are count / ``vessel_count_total``. If the typed counts do not add
    up to the total within ``tolerance`` a warning is issued (not an error:
    PortWatch also has an "other" class); ``other`` holds the remainder
    (floored at 0).
    """
    _need(info, ["chokepoint", "vessel_count_total", *TYPE_COLUMNS.values()])
    total = info["vessel_count_total"].astype("float64")
    out = pd.DataFrame({"chokepoint": info["chokepoint"].to_numpy()})
    for label, col in TYPE_COLUMNS.items():
        out[label] = (info[col].astype("float64") / total).to_numpy()
    typed = out[list(TYPE_COLUMNS)].sum(axis=1, skipna=True)
    out["other"] = (1.0 - typed).clip(lower=0.0)
    over = [c for c, t in zip(out["chokepoint"], typed) if t > 1 + tolerance]
    if over:
        warnings.warn(f"typed vessel counts exceed the total for {over}", stacklevel=2)
    return out.sort_values("chokepoint").reset_index(drop=True)


def _norm(label: object) -> str:
    return str(label).strip().lower()


def industry_ranks(
    info: pd.DataFrame,
    industry_to_sector: Mapping[str, str],
    rank_weights: tuple[float, ...] = DEFAULT_RANK_WEIGHTS,
) -> pd.DataFrame:
    """Long table: ``chokepoint, rank, industry, sector, weight``.

    ``sector`` is NaN for industries missing from the mapping; use
    ``unmapped_industries`` to see which.
    """
    _need(info, ["chokepoint", *RANK_COLUMNS])
    if len(rank_weights) != len(RANK_COLUMNS):
        raise ValueError("need one weight per rank")
    rows = []
    for _, r in info.iterrows():
        for rank, (col, w) in enumerate(zip(RANK_COLUMNS, rank_weights), start=1):
            ind = r[col]
            if pd.isna(ind) or str(ind).strip() == "":
                continue
            rows.append(
                {
                    "chokepoint": r["chokepoint"],
                    "rank": rank,
                    "industry": str(ind).strip(),
                    "sector": industry_to_sector.get(_norm(ind)),
                    "weight": w,
                }
            )
    return pd.DataFrame(
        rows, columns=["chokepoint", "rank", "industry", "sector", "weight"]
    )


def unmapped_industries(ranks: pd.DataFrame) -> list[str]:
    """Industry labels in PortWatch that the mapping does not cover."""
    return sorted(ranks.loc[ranks["sector"].isna(), "industry"].unique())


def sector_exposure(
    info: pd.DataFrame,
    industry_to_sector: Mapping[str, str],
    rank_weights: tuple[float, ...] = DEFAULT_RANK_WEIGHTS,
) -> pd.DataFrame:
    """Ordinal sector score per chokepoint: ``chokepoint, sector, score``.

    ``score`` is the sum of the rank weights of the industries mapped to the
    sector, divided by the sum of all weights (so a sector holding all three
    ranks scores 1.0). It orders sectors within a chokepoint; it is not a
    share of trade volume.
    """
    ranks = industry_ranks(info, industry_to_sector, rank_weights)
    mapped = ranks.dropna(subset=["sector"])
    denom = float(sum(rank_weights))
    out = (
        mapped.groupby(["chokepoint", "sector"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "score"})
    )
    out["score"] = out["score"] / denom
    return out.sort_values(
        ["chokepoint", "score", "sector"], ascending=[True, False, True]
    ).reset_index(drop=True)


def chokepoints_for_sector(
    exposure: pd.DataFrame, sector: str, top: int = 5
) -> pd.DataFrame:
    """Chokepoints where a sector scores highest."""
    sub = exposure[exposure["sector"] == sector]
    return (
        sub.sort_values(["score", "chokepoint"], ascending=[False, True])
        .head(top)
        .reset_index(drop=True)
    )


def proxies_for(
    chokepoint: str,
    exposure: pd.DataFrame,
    proxies: Mapping[str, Mapping],
    *,
    top_sectors: int = 2,
) -> pd.DataFrame:
    """Tickers to watch for a chokepoint: ``ticker, reason, sector``.

    Route-specific names come first, then proxies of the chokepoint's top
    ``top_sectors`` sectors. Duplicates keep their first reason.
    """
    rows = [
        {"ticker": t, "reason": "route", "sector": None}
        for t in proxies.get("chokepoint_proxies", {}).get(chokepoint, [])
    ]
    sub = exposure[exposure["chokepoint"] == chokepoint].head(top_sectors)
    for sector in sub["sector"]:
        for t in proxies.get("sector_proxies", {}).get(sector, []):
            rows.append({"ticker": t, "reason": "sector", "sector": sector})
    df = pd.DataFrame(rows, columns=["ticker", "reason", "sector"])
    return df.drop_duplicates("ticker").reset_index(drop=True)


def lookup(
    chokepoint: str,
    info: pd.DataFrame,
    proxies: Mapping[str, Mapping],
) -> dict[str, object]:
    """One-stop exposure card for a chokepoint."""
    if chokepoint not in set(info["chokepoint"]):
        raise KeyError(f"unknown chokepoint: {chokepoint}")
    exposure = sector_exposure(info, proxies["industry_to_sector"])
    shares = vessel_type_shares(info)
    return {
        "chokepoint": chokepoint,
        "vessel_type_shares": shares[shares["chokepoint"] == chokepoint]
        .drop(columns="chokepoint")
        .iloc[0]
        .to_dict(),
        "sectors": exposure[exposure["chokepoint"] == chokepoint].reset_index(
            drop=True
        ),
        "proxies": proxies_for(chokepoint, exposure, proxies),
    }
