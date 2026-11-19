import warnings

import pandas as pd
import pytest

from openbb_chokepoints.core import exposure as ex

MAP = {"crude oil": "energy", "vehicles": "autos", "cereals": "agriculture"}
PROXIES = {
    "industry_to_sector": MAP,
    "sector_proxies": {"energy": ["XLE"], "autos": ["TM"], "agriculture": ["DBA"]},
    "chokepoint_proxies": {"strait_of_hormuz": ["FRO", "XLE"]},
}


@pytest.fixture
def info():
    return pd.DataFrame(
        {
            "chokepoint": ["strait_of_hormuz", "suez_canal"],
            "vessel_count_total": [100, 200],
            "vessel_count_tanker": [60, 20],
            "vessel_count_container": [10, 100],
            "vessel_count_general_cargo": [10, 20],
            "vessel_count_dry_bulk": [10, 40],
            "vessel_count_roro": [5, 10],
            "industry_top1": ["Crude Oil ", "Vehicles"],
            "industry_top2": ["Vehicles", "Mystery Goods"],
            "industry_top3": [None, "Cereals"],
        }
    )


def test_type_shares(info):
    sh = ex.vessel_type_shares(info).set_index("chokepoint")
    assert sh.loc["strait_of_hormuz", "tanker"] == pytest.approx(0.6)
    assert sh.loc["suez_canal", "container"] == pytest.approx(0.5)
    assert sh.loc["strait_of_hormuz", "other"] == pytest.approx(0.05)


def test_type_shares_warn_when_over_total(info):
    info.loc[0, "vessel_count_tanker"] = 500
    with pytest.warns(UserWarning, match="exceed"):
        ex.vessel_type_shares(info)


def test_missing_columns(info):
    with pytest.raises(ValueError, match="missing columns"):
        ex.vessel_type_shares(info.drop(columns="vessel_count_roro"))
    with pytest.raises(ValueError):
        ex.industry_ranks(info.drop(columns="industry_top2"), MAP)


def test_ranks_normalise_labels_and_skip_blanks(info):
    r = ex.industry_ranks(info, MAP)
    assert len(r) == 5  # one blank skipped
    assert (
        r.loc[
            (r["chokepoint"] == "strait_of_hormuz") & (r["rank"] == 1), "sector"
        ].iloc[0]
        == "energy"
    )
    assert ex.unmapped_industries(r) == ["Mystery Goods"]
    with pytest.raises(ValueError):
        ex.industry_ranks(info, MAP, rank_weights=(1.0,))


def test_sector_exposure_is_ordinal_and_bounded(info):
    e = ex.sector_exposure(info, MAP)
    h = e[e["chokepoint"] == "strait_of_hormuz"].set_index("sector")["score"]
    assert h["energy"] == pytest.approx(3 / 6)
    assert h["autos"] == pytest.approx(2 / 6)
    assert (e["score"] <= 1).all()
    assert list(e[e["chokepoint"] == "strait_of_hormuz"]["sector"]) == [
        "energy",
        "autos",
    ]


def test_chokepoints_for_sector(info):
    e = ex.sector_exposure(info, MAP)
    out = ex.chokepoints_for_sector(e, "autos")
    assert list(out["chokepoint"]) == ["suez_canal", "strait_of_hormuz"]


def test_proxies_dedupe_and_order(info):
    e = ex.sector_exposure(info, MAP)
    p = ex.proxies_for("strait_of_hormuz", e, PROXIES)
    assert list(p["ticker"]) == ["FRO", "XLE", "TM"]
    assert p.loc[p["ticker"] == "XLE", "reason"].iloc[0] == "route"


def test_lookup(info):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        card = ex.lookup("suez_canal", info, PROXIES)
    assert card["vessel_type_shares"]["container"] == pytest.approx(0.5)
    assert "TM" in set(card["proxies"]["ticker"])
    with pytest.raises(KeyError):
        ex.lookup("narnia", info, PROXIES)
