"""Detection validation against the event catalog (WP5). Pure pandas/numpy.

Definitions (fixed before looking at results)
---------------------------------------------
* Targets are catalog events with ``type == "disruption"``.
* An event is a **hit** if the monitor has an episode on the event's primary
  chokepoint whose alert day (``start``) lies in
  ``[news_date - pre_days, news_date + post_days]`` (hit_type ``"new"``), or
  an episode already active on ``news_date`` (``onset <= news_date <= end``,
  hit_type ``"covered"``: typical for overlapping Red Sea events).
* **Delay** = ``start - news_date`` in days for "new" hits (negative = the
  monitor led the news; 0 for "covered").
* An episode is a **false alarm** if its alert day lies in no event window of
  a chokepoint it is attributed to. An event's window is
  ``[news_date - pre_days, max(news_date + fa_days, end_date)]`` and applies to
  the event's primary AND related chokepoints (a documented loosening: an
  upstream route disruption legitimately moves neighbours).
* recall = hits / events; precision = 1 - false_alarms / episodes;
  F1 = harmonic mean.
* Scope: by default only chokepoints named in the catalog are scored, since
  episodes elsewhere may be real but uncatalogued.

Grid and selection rule: 18 combinations (3 tau x 3 (k, n) x 2 smoothing).
Choose on the selection period by maximum F1; ties -> fewer false alarms ->
larger tau -> larger n. Report the winner on the held-out period and on the
full period. Selection on very few events is fragile; ``select_params`` warns
when the selection period holds fewer than ``min_select_events`` events.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openbb_chokepoints.core.detection import (
    DetectionParams,
    apply_rule,
    compute_scores,
    extract_episodes,
)


@dataclass(frozen=True)
class Windows:
    pre_days: int = 3
    post_days: int = 14
    fa_days: int = 60


def default_grid(base: DetectionParams | None = None) -> list[DetectionParams]:
    """The 18-combination grid."""
    b = base or DetectionParams()
    return [
        b.with_(tau=tau, k=k, n=n, smooth_days=sm)
        for sm, tau, (k, n) in itertools.product(
            (3, 7), (2.5, 3.0, 3.5), ((2, 3), (3, 5), (4, 7))
        )
    ]


def _event_windows(events: pd.DataFrame, w: Windows) -> pd.DataFrame:
    rows = []
    for _, e in events.iterrows():
        start = e["news_date"] - pd.Timedelta(days=w.pre_days)
        end = e["news_date"] + pd.Timedelta(days=w.fa_days)
        if pd.notna(e.get("end_date")):
            end = max(end, e["end_date"])
        for cp in [e["chokepoint"], *list(e["related_chokepoints"])]:
            rows.append({"event_id": e["id"], "chokepoint": cp, "lo": start, "hi": end})
    return pd.DataFrame(rows, columns=["event_id", "chokepoint", "lo", "hi"])


def score_events(
    events: pd.DataFrame, episodes: pd.DataFrame, windows: Windows | None = None
) -> pd.DataFrame:
    """Per-event result: ``event_id, hit, hit_type, delay_days, episode_start``."""
    w = windows or Windows()
    rows = []
    disruptions = events[events["type"] == "disruption"]
    for _, e in disruptions.iterrows():
        eps = episodes[episodes["chokepoint"] == e["chokepoint"]]
        news = e["news_date"]
        covered = eps[(eps["onset"] <= news) & (eps["end"] >= news)]
        lo = news - pd.Timedelta(days=w.pre_days)
        hi = news + pd.Timedelta(days=w.post_days)
        new = eps[(eps["start"] >= lo) & (eps["start"] <= hi)].sort_values("start")
        if len(new):
            row = new.iloc[0]
            rows.append(
                (e["id"], True, "new", (row["start"] - news).days, row["start"])
            )
        elif len(covered):
            rows.append((e["id"], True, "covered", 0, covered.iloc[0]["start"]))
        else:
            rows.append((e["id"], False, "miss", np.nan, pd.NaT))
    return pd.DataFrame(
        rows, columns=["event_id", "hit", "hit_type", "delay_days", "episode_start"]
    )


def label_episodes(
    events: pd.DataFrame, episodes: pd.DataFrame, windows: Windows | None = None
) -> pd.DataFrame:
    """Add ``false_alarm`` (bool) to each episode."""
    w = windows or Windows()
    win = _event_windows(events[events["type"] == "disruption"], w)
    flags = []
    for _, ep in episodes.iterrows():
        sub = win[win["chokepoint"] == ep["chokepoint"]]
        ok = ((sub["lo"] <= ep["start"]) & (ep["start"] <= sub["hi"])).any()
        flags.append(not ok)
    out = episodes.copy()
    out["false_alarm"] = flags
    return out


def metrics(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    windows: Windows | None = None,
    years: float | None = None,
) -> dict[str, float]:
    """Metrics for events with ``news_date`` and episodes with ``start`` in range."""
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    ev = events[(events["news_date"] >= lo) & (events["news_date"] <= hi)]
    ep = episodes[(episodes["start"] >= lo) & (episodes["start"] <= hi)]
    # windows of events just outside the range still excuse episodes inside it
    labelled = label_episodes(events, ep, windows)
    scored = score_events(ev, episodes, windows)
    n_events = len(scored)
    hits = int(scored["hit"].sum()) if n_events else 0
    fa = int(labelled["false_alarm"].sum()) if len(labelled) else 0
    n_ep = len(labelled)
    recall = hits / n_events if n_events else np.nan
    precision = 1 - fa / n_ep if n_ep else np.nan
    if np.isnan(recall) or np.isnan(precision) or recall + precision == 0:
        f1 = np.nan
    else:
        f1 = 2 * recall * precision / (recall + precision)
    delays = scored.loc[scored["hit_type"] == "new", "delay_days"]
    return {
        "n_events": n_events,
        "hits": hits,
        "recall": recall,
        "n_episodes": n_ep,
        "false_alarms": fa,
        "precision": precision,
        "f1": f1,
        "median_delay_days": float(delays.median()) if len(delays) else np.nan,
        "fa_per_year": fa / years if years else np.nan,
    }


def _scope(events: pd.DataFrame) -> list[str]:
    cps: set[str] = set()
    for _, e in events.iterrows():
        cps.add(e["chokepoint"])
        cps.update(e["related_chokepoints"])
    return sorted(cps)


def _volume_by_chokepoint(
    volume: pd.DataFrame, chokepoints: Iterable[str], vessel_type: str, metric: str
) -> dict[str, pd.Series]:
    sub = volume[
        (volume["vessel_type"] == vessel_type) & volume["chokepoint"].isin(chokepoints)
    ]
    return {name: g.set_index("date")[metric] for name, g in sub.groupby("chokepoint")}


def run_grid(
    volume: pd.DataFrame,
    events: pd.DataFrame,
    grid: list[DetectionParams] | None = None,
    *,
    periods: dict[str, tuple[str | None, str | None]] | None = None,
    windows: Windows | None = None,
    vessel_type: str = "total",
    metric: str = "vessels",
    chokepoints: list[str] | None = None,
) -> pd.DataFrame:
    """Score every grid point on every period.

    Returns one row per (grid point, period) with the parameters and metrics.
    Default periods: ``select`` 2019-2022, ``report`` 2023-2026, ``full``.
    """
    grid = grid or default_grid()
    periods = periods or {
        "select": ("2019-01-01", "2022-12-31"),
        "report": ("2023-01-01", "2026-12-31"),
        "full": (None, None),
    }
    scope = chokepoints or _scope(events)
    series = _volume_by_chokepoint(volume, scope, vessel_type, metric)
    if not series:
        raise ValueError("no volume data for the catalogued chokepoints")
    span = volume["date"]
    years = (span.max() - span.min()).days / 365.25

    score_cache: dict[tuple[int, int, int, float], dict[str, pd.DataFrame]] = {}
    rows = []
    for params in grid:
        key = (
            params.smooth_days,
            params.baseline_days,
            params.baseline_lag,
            params.min_scale_frac,
        )
        if key not in score_cache:
            score_cache[key] = {
                name: compute_scores(s, params) for name, s in series.items()
            }
        parts = []
        for name, sc in score_cache[key].items():
            eps = extract_episodes(apply_rule(sc, params), params)
            if len(eps):
                parts.append(eps.assign(chokepoint=name))
        episodes = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["chokepoint", "onset", "start", "end"])
        )
        for label, (a, b) in periods.items():
            if label == "full":
                yrs = years
            else:
                lo = pd.Timestamp(a) if a else span.min()
                hi = pd.Timestamp(b) if b else span.max()
                yrs = max((min(hi, span.max()) - max(lo, span.min())).days, 1) / 365.25
            m = metrics(events, episodes, start=a, end=b, windows=windows, years=yrs)
            rows.append({"period": label, **params.as_dict(), **m})
    return pd.DataFrame(rows)


def select_params(
    results: pd.DataFrame, *, period: str = "select", min_select_events: int = 3
) -> pd.Series:
    """Apply the pre-declared selection rule; returns the winning row (on
    the selection period)."""
    sel = results[results["period"] == period]
    if sel.empty:
        raise ValueError(f"no results for period '{period}'")
    if sel["n_events"].iloc[0] < min_select_events:
        warnings.warn(
            f"selection period has only {int(sel['n_events'].iloc[0])} events; "
            "the choice is fragile",
            stacklevel=2,
        )
    sel = sel.assign(_f1=sel["f1"].fillna(-1.0))
    sel = sel.sort_values(
        ["_f1", "false_alarms", "tau", "n"], ascending=[False, True, False, False]
    )
    return sel.iloc[0].drop("_f1")


def cusum_episodes(
    scores: pd.DataFrame, *, drift: float = 0.5, threshold: float = 5.0
) -> pd.DataFrame:
    """Comparator detector: one-sided CUSUM on the robust z-scores.

    ``S_t = min(0, S_{t-1} + z_t + drift)``; flagged when ``S_t <= -threshold``.
    Returns episodes in the same format as the main monitor.
    """
    z = scores["z"].fillna(0.0).to_numpy()
    s = np.zeros(len(z))
    for i in range(len(z)):
        prev = s[i - 1] if i else 0.0
        s[i] = min(0.0, prev + z[i] + drift)
    alt = scores.copy()
    alt["flag"] = s <= -threshold
    alt["low"] = scores["z"] <= -3.0
    return extract_episodes(alt, DetectionParams(merge_gap=3))
