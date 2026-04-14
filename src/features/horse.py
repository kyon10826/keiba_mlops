"""Horse-level rolling performance features.

All rolling aggregations use shift(1) to prevent data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rolling features that benefit from cold-start imputation.
# When a horse has no prior races, these are 0 or NaN; filling with
# the sire-average gives the model a reasonable prior.
COLD_START_FILL_FEATURES = [
    "rank_last",
    "rank_rolling_3",
    "rank_rolling_5",
    "show_rate_last_5",
    "last_3f_rolling_3",
    "time_diff_rolling_3",
]

# Features left at 0 for first-timers (0 is semantically correct).
# race_span_days=0  → no previous race
# prize_cumsum=0    → no earnings yet
# label_momentum=0  → no trend data
_COLD_START_SKIP = {"race_span_days", "prize_cumsum", "label_momentum"}


def compute_cold_start_defaults(
    train_df: pd.DataFrame,
    sire_stats_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-sire average rolling features from training data.

    Only rows with *actual* prior-race history are used (rows where
    ``rank_last > 0``), so first-race filler values are excluded.

    Args:
        train_df: Training dataframe **after** ``add_horse_features()``
            has been applied (must contain the rolling columns).
        sire_stats_df: Unused, reserved for future extensions.

    Returns:
        ``{father_name: {feature: mean_value, ...}, ...}``
        A special key ``"_GLOBAL_"`` holds the overall average used as
        a fallback when the sire is unknown.
    """
    if "rank_last" not in train_df.columns:
        return {"_GLOBAL_": {f: 0.0 for f in COLD_START_FILL_FEATURES}}

    # Keep only rows with genuine prior-race data.
    valid = train_df[train_df["rank_last"] > 0].copy()
    if valid.empty:
        return {"_GLOBAL_": {f: 0.0 for f in COLD_START_FILL_FEATURES}}

    valid["father"] = valid["father"].astype(str)

    # Global fallback
    global_avgs: dict[str, float] = {}
    for feat in COLD_START_FILL_FEATURES:
        if feat in valid.columns:
            global_avgs[feat] = float(valid[feat].mean())
        else:
            global_avgs[feat] = 0.0

    # Per-sire averages
    defaults: dict[str, dict[str, float]] = {"_GLOBAL_": global_avgs}
    for father, grp in valid.groupby("father"):
        avgs: dict[str, float] = {}
        for feat in COLD_START_FILL_FEATURES:
            if feat in grp.columns:
                avgs[feat] = float(grp[feat].mean())
            else:
                avgs[feat] = 0.0
        defaults[str(father)] = avgs

    return defaults


def apply_cold_start_defaults(
    df: pd.DataFrame,
    defaults: dict[str, dict[str, float]],
    father_col: str = "father",
) -> pd.DataFrame:
    """Fill cold-start rolling features using sire-based defaults.

    A row is considered "cold start" when ``rank_last`` is 0 or NaN
    (meaning the horse has no prior race record in the dataset).

    For each such row the function looks up the horse's sire in
    *defaults* and replaces 0/NaN feature values with the sire average.
    If the sire is not found, the ``"_GLOBAL_"`` fallback is used.

    Args:
        df: DataFrame with rolling features already computed.
        defaults: Output of :func:`compute_cold_start_defaults`.
        father_col: Column name for the sire.

    Returns:
        DataFrame with cold-start features filled.
    """
    out = df.copy()

    cold_mask = (out["rank_last"] == 0) | (out["rank_last"].isna())
    if not cold_mask.any():
        return out

    out[father_col] = out[father_col].astype(str)
    global_defaults = defaults.get("_GLOBAL_", {})
    cold_idx = out[cold_mask].index
    fathers = out.loc[cold_idx, father_col]

    for feat in COLD_START_FILL_FEATURES:
        if feat not in out.columns:
            continue
        # Build per-row default values from the sire lookup.
        fill_values = fathers.map(
            lambda f, _feat=feat: defaults.get(f, global_defaults).get(_feat, 0.0)
        )
        # Only overwrite where the current value is 0 or NaN.
        needs_fill = cold_mask & ((out[feat] == 0) | (out[feat].isna()))
        out.loc[needs_fill, feat] = fill_values.reindex(out.loc[needs_fill].index)

    return out


def add_horse_features(
    df: pd.DataFrame,
    cold_start_defaults: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Add per-horse rolling performance features.

    The dataframe must be sorted by (id, race_id) before calling.

    Args:
        df: Input dataframe.
        cold_start_defaults: If provided, cold-start horses (rank_last==0)
            will have their rolling features filled with sire-based averages
            from this dict (output of :func:`compute_cold_start_defaults`).
            When ``None`` (default), behaviour is identical to the original
            implementation.

    Features created:
        rank_last: Previous race rank
        rank_rolling_3: Mean rank over last 3 races
        rank_rolling_5: Mean rank over last 5 races
        show_rate_last_5: Show rate (1-3 finish) over last 5 races
        last_3f_rolling_3: Mean last-3F time over last 3 races
        time_diff_rolling_3: Mean time diff over last 3 races
        weight_horse: Horse weight (passthrough, renamed for clarity)
        weight_change: Weight change (inc_dec)
        race_span_days: Days since previous race
        prize_cumsum: Cumulative prize earnings
        label_momentum: Change in rank over last 2 races
    """
    out = df.sort_values(["id", "race_id"]).copy()
    g = out.groupby("id")

    # Shifted rank (leak-safe)
    out["rank_last"] = g["rank"].shift(1).fillna(0)

    # Rolling mean rank (grouped to prevent cross-horse contamination)
    shifted_rank = g["rank"].shift(1)
    out["rank_rolling_3"] = shifted_rank.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["rank_rolling_5"] = shifted_rank.groupby(out["id"]).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # Show flag per race (1-3 finish)
    out["_show_flag"] = ((out["rank"] >= 1) & (out["rank"] <= 3)).astype(float)
    shifted_show = g["_show_flag"].shift(1)
    out["show_rate_last_5"] = shifted_show.groupby(out["id"]).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # Last 3F rolling
    shifted_3f = g["last_3F_time"].shift(1)
    out["last_3f_rolling_3"] = shifted_3f.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Time diff rolling
    out["_time_diff_num"] = pd.to_numeric(out["time_diff"], errors="coerce")
    shifted_td = g["_time_diff_num"].shift(1)
    out["time_diff_rolling_3"] = shifted_td.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Weight passthrough
    out["weight_horse"] = out["weight"].astype(float)

    # Weight change
    out["weight_change"] = out["inc_dec"].astype(float).fillna(0)

    # Race span in days (approximate: use year/month/day)
    out["_race_date"] = pd.to_datetime(
        out["year"].astype(str).apply(lambda y: ("20" + y) if len(y) <= 2 else y)
        + "-" + out["month"].astype(str).str.zfill(2)
        + "-" + out["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    out["race_span_days"] = g["_race_date"].diff().dt.days.fillna(0)

    # Cumulative prize (shifted to prevent leakage)
    out["prize_cumsum"] = g["prize"].apply(lambda s: s.shift(1).cumsum().fillna(0)).values

    # Label momentum: rank change over last 2 races
    rank_shift1 = g["rank"].shift(1)
    rank_shift2 = g["rank"].shift(2)
    out["label_momentum"] = (rank_shift1 - rank_shift2).fillna(0)

    # Clean up temp columns
    out.drop(columns=["_show_flag", "_time_diff_num", "_race_date"], inplace=True)

    # Apply cold-start defaults when provided (no-op for training).
    if cold_start_defaults is not None:
        out = apply_cold_start_defaults(out, cold_start_defaults)

    return out
