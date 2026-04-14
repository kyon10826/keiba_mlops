"""レース内の相対特徴量。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """同一レース内の他の出走馬との相対値から算出する特徴量を追加する。

    レースのグルーピングキー: (year, month, day, place, race_num)

    作成される特徴量:
        odds_rank: レース内における win_odds の順位(オッズが低いほど順位が小さい)
        weight_zscore: レース内での馬体重の Z スコア
        age_relative: レース内平均年齢との差
    """
    out = df.copy()
    race_key = ["year", "month", "day", "place", "race_num"]

    # オッズ順位
    out["odds_rank"] = out.groupby(race_key)["win_odds"].rank(method="min").fillna(0)

    # 馬体重の Z スコア
    race_weight_mean = out.groupby(race_key)["weight"].transform("mean")
    race_weight_std = out.groupby(race_key)["weight"].transform("std").replace(0, 1)
    out["weight_zscore"] = ((out["weight"].astype(float) - race_weight_mean) / race_weight_std).fillna(0)

    # 相対年齢
    race_age_mean = out.groupby(race_key)["age"].transform("mean")
    out["age_relative"] = (out["age"].astype(float) - race_age_mean).fillna(0)

    return out
