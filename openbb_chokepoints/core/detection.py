"""Disruption monitor (WP3). Pure pandas/numpy: imports nothing from OpenBB.

Method
------
1. Reindex the series to a full daily calendar (missing days stay NaN).
2. Smooth with a trailing mean (``smooth_days``).
3. Baseline for day t = median and MAD of the smoothed series over the
   ``baseline_days`` ending ``baseline_lag`` days before t (so a disruption
   cannot contaminate its own baseline immediately).
4. Robust z = (smoothed - median) / scale, with
   scale = max(1.4826 * MAD, min_scale_frac * median, abs_floor).
5. A day is *low* if z <= -tau. A day is *flagged* if at least ``k`` of the
   last ``n`` days are low.
6. Consecutive flagged runs separated by <= ``merge_gap`` days are merged into
   one episode.

Year-on-year change uses a 364-day shift (same weekday).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826

DAILY_COLUMNS = [
    "value",
    "smoothed",
    "baseline_median",
    "baseline_scale",
    "z",
    "yoy",
    "low",
    "flag",
]
EPISODE_COLUMNS = [
    "onset",
    "start",
    "end",
    "duration_days",
    "min_z",
    "peak_drop_pct",
    "ongoing",
]


@dataclass(frozen=True)
class DetectionParams:
    smooth_days: int = 7
    baseline_days: int = 90
    baseline_lag: int = 7
    min_baseline_obs: int = 45
    tau: float = 3.0
    k: int = 3
    n: int = 5
    merge_gap: int = 3
    min_scale_frac: float = 0.05
    abs_floor: float = 1.0

    def __post_init__(self) -> None:
        if self.smooth_days < 1 or self.baseline_days < 2 or self.baseline_lag < 0:
            raise ValueError("invalid window lengths")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if not 1 <= self.k <= self.n:
            raise ValueError("need 1 <= k <= n")
        if self.min_baseline_obs > self.baseline_days:
            raise ValueError("min_baseline_obs cannot exceed baseline_days")
        if self.merge_gap < 0:
            raise ValueError("merge_gap must be >= 0")

    def with_(self, **kwargs: float) -> DetectionParams:
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return asdict(self)


def _daily(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    s = series.astype("float64").sort_index(kind="stable")
    s = s[~s.index.duplicated(keep="last")]
    s.index = s.index.normalize()
    if s.empty:
        return s
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


def _mad(window: np.ndarray) -> float:
    med = np.nanmedian(window)
    return float(np.nanmedian(np.abs(window - med)))


def _rule(z: pd.Series, p: DetectionParams) -> tuple[pd.Series, pd.Series]:
    low = (z <= -p.tau).fillna(False)
    recent_low = low.astype(float).rolling(p.n, min_periods=1).sum()
    return low, recent_low >= p.k


def apply_rule(scores: pd.DataFrame, params: DetectionParams) -> pd.DataFrame:
    """Recompute ``low``/``flag`` from existing z-scores with new tau/k/n.

    The z-scores depend only on the smoothing and baseline settings, so a
    parameter grid over tau/k/n can reuse them.
    """
    out = scores.copy()
    out["low"], out["flag"] = _rule(out["z"], params)
    return out


def compute_scores(
    series: pd.Series, params: DetectionParams | None = None
) -> pd.DataFrame:
    """Daily scores for one series. Columns: ``DAILY_COLUMNS``."""
    p = params or DetectionParams()
    s = _daily(series)
    smoothed = s.rolling(p.smooth_days, min_periods=max(1, p.smooth_days // 2)).mean()
    roll = smoothed.rolling(p.baseline_days, min_periods=p.min_baseline_obs)
    median = roll.median().shift(p.baseline_lag)
    mad = roll.apply(_mad, raw=True).shift(p.baseline_lag)
    scale = np.maximum.reduce(
        [
            (MAD_TO_SIGMA * mad).to_numpy(),
            (p.min_scale_frac * median).to_numpy(),
            np.full(len(s), p.abs_floor),
        ]
    )
    scale = pd.Series(scale, index=s.index).where(median.notna())
    z = (smoothed - median) / scale
    yoy = smoothed / smoothed.shift(364) - 1.0
    low, flag = _rule(z, p)
    out = pd.DataFrame(
        {
            "value": s,
            "smoothed": smoothed,
            "baseline_median": median,
            "baseline_scale": scale,
            "z": z,
            "yoy": yoy,
            "low": low,
            "flag": flag,
        }
    )
    return out[DAILY_COLUMNS]


def extract_episodes(
    scores: pd.DataFrame, params: DetectionParams | None = None
) -> pd.DataFrame:
    """Merge flagged days into episodes.

    ``start`` is the first flagged day (when the monitor would have alerted),
    ``onset`` the first low day inside the confirming window (the likely true
    start), ``end`` the last flagged day. ``ongoing`` is True when the episode
    reaches the final observed day.
    """
    p = params or DetectionParams()
    flagged = scores.index[scores["flag"].to_numpy()]
    if len(flagged) == 0:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    runs: list[list[pd.Timestamp]] = [[flagged[0], flagged[0]]]
    for day in flagged[1:]:
        if (day - runs[-1][1]).days <= p.merge_gap + 1:
            runs[-1][1] = day
        else:
            runs.append([day, day])
    last_day = scores.index[-1]
    rows = []
    for start, end in runs:
        window = scores.loc[start - pd.Timedelta(days=p.n - 1) : start]
        low_days = window.index[window["low"].to_numpy()]
        onset = low_days[0] if len(low_days) else start
        seg = scores.loc[onset:end]
        base = seg["baseline_median"]
        drop = ((seg["smoothed"] / base) - 1.0).min()
        rows.append(
            {
                "onset": onset,
                "start": start,
                "end": end,
                "duration_days": (end - onset).days + 1,
                "min_z": float(seg["z"].min()),
                "peak_drop_pct": float(drop * 100.0),
                "ongoing": bool(end == last_day),
            }
        )
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def detect(
    series: pd.Series, params: DetectionParams | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(daily_scores, episodes)`` for one series."""
    p = params or DetectionParams()
    scores = compute_scores(series, p)
    return scores, extract_episodes(scores, p)


def detect_all(
    volume: pd.DataFrame,
    params: DetectionParams | None = None,
    *,
    vessel_type: str = "total",
    metric: str = "vessels",
) -> pd.DataFrame:
    """Run the monitor on every chokepoint in a long volume table.

    ``volume`` has the WP1 columns (``date, chokepoint, vessel_type, vessels,
    capacity_tons``). Returns all episodes with a ``chokepoint`` column.
    """
    p = params or DetectionParams()
    need = {"date", "chokepoint", "vessel_type", metric}
    if not need.issubset(volume.columns):
        raise ValueError(f"volume table must have columns {sorted(need)}")
    sub = volume[volume["vessel_type"] == vessel_type]
    parts = []
    for name, grp in sub.groupby("chokepoint", sort=True):
        _, eps = detect(grp.set_index("date")[metric], p)
        if len(eps):
            parts.append(eps.assign(chokepoint=name))
    if not parts:
        return pd.DataFrame(columns=["chokepoint", *EPISODE_COLUMNS])
    out = pd.concat(parts, ignore_index=True)
    return out[["chokepoint", *EPISODE_COLUMNS]].sort_values(
        ["start", "chokepoint"], ignore_index=True
    )


def current_status(
    series: pd.Series, params: DetectionParams | None = None
) -> dict[str, object]:
    """Latest reading for dashboards: z, yoy, flag, and data date."""
    scores = compute_scores(series, params)
    valid = scores.dropna(subset=["value"])
    if valid.empty:
        return {"date": None, "z": np.nan, "yoy": np.nan, "flag": False}
    row = valid.iloc[-1]
    return {
        "date": valid.index[-1],
        "z": float(row["z"]) if pd.notna(row["z"]) else np.nan,
        "yoy": float(row["yoy"]) if pd.notna(row["yoy"]) else np.nan,
        "flag": bool(row["flag"]),
    }
