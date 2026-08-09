"""レース内の相対特徴量。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """同一レース内の他の出走馬との相対値から算出する特徴量を追加する。

    レースのグルーピングキー: (year, month, day, place, race_num)

    作成される特徴量:
        weight_zscore:      レース内での馬体重の Z スコア
        age_relative:       レース内平均年齢との差
        basis_weight_zscore:レース内での斤量の Z スコア
        prize_zscore:       レース内での累積賞金の Z スコア (相対的な格の指標)
        last_3f_relative:   レース内での前走上がり 3F 平均との差 (相対的スピード指標)

    ★ 注意: odds_rank は市場情報 (win_odds) 由来なので、
       AI 競馬予想マスターズのオッズ不使用方針に沿って削除している。
    """
    out = df.copy()
    race_key = ["year", "month", "day", "place", "race_num"]

    # 馬体重の Z スコア
    race_weight_mean = out.groupby(race_key)["weight"].transform("mean")
    race_weight_std = out.groupby(race_key)["weight"].transform("std").replace(0, 1)
    out["weight_zscore"] = ((out["weight"].astype(float) - race_weight_mean) / race_weight_std).fillna(0)

    # 相対年齢
    race_age_mean = out.groupby(race_key)["age"].transform("mean")
    out["age_relative"] = (out["age"].astype(float) - race_age_mean).fillna(0)

    # 斤量の Z スコア (ハンデ戦での相対的な有利不利を捉える)
    if "basis_weight" in out.columns:
        race_bw_mean = out.groupby(race_key)["basis_weight"].transform("mean")
        race_bw_std = out.groupby(race_key)["basis_weight"].transform("std").replace(0, 1)
        out["basis_weight_zscore"] = (
            (out["basis_weight"].astype(float) - race_bw_mean) / race_bw_std
        ).fillna(0)
    else:
        out["basis_weight_zscore"] = 0.0

    # 累積賞金の Z スコア (レース内での「格」の相対値)
    if "prize_cumsum" in out.columns:
        race_pz_mean = out.groupby(race_key)["prize_cumsum"].transform("mean")
        race_pz_std = out.groupby(race_key)["prize_cumsum"].transform("std").replace(0, 1)
        out["prize_zscore"] = (
            (out["prize_cumsum"].astype(float) - race_pz_mean) / race_pz_std
        ).fillna(0)
    else:
        out["prize_zscore"] = 0.0

    # 前走上がり 3F の相対値 (レース内の相対的なスピード持ちを表現)
    if "last_3f_rolling_3" in out.columns:
        race_3f_mean = out.groupby(race_key)["last_3f_rolling_3"].transform("mean")
        out["last_3f_relative"] = (
            out["last_3f_rolling_3"].astype(float) - race_3f_mean
        ).fillna(0)
    else:
        out["last_3f_relative"] = 0.0

    return out
