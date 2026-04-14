"""Within-race relative features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computed relative to other runners in the same race.

    Race grouping key: (year, month, day, place, race_num)

    Features created:
        odds_rank: Rank of win_odds within the race (lower odds = lower rank)
        weight_zscore: Z-score of horse weight within the race
        age_relative: Difference from mean age within the race
    """
    out = df.copy()
    race_key = ["year", "month", "day", "place", "race_num"]

    # odds_rank
    out["odds_rank"] = out.groupby(race_key)["win_odds"].rank(method="min").fillna(0)

    # weight_zscore
    race_weight_mean = out.groupby(race_key)["weight"].transform("mean")
    race_weight_std = out.groupby(race_key)["weight"].transform("std").replace(0, 1)
    out["weight_zscore"] = ((out["weight"].astype(float) - race_weight_mean) / race_weight_std).fillna(0)

    # age_relative
    race_age_mean = out.groupby(race_key)["age"].transform("mean")
    out["age_relative"] = (out["age"].astype(float) - race_age_mean).fillna(0)

    return out
