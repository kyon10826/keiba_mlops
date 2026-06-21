"""時系列オッズ特徴量を main DataFrame に結合するユーティリティ。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.time_series_odds import (
    derive_ts_race_ids_from_main,
    load_jd_for_years,
)

TS_ODDS_FEATURE_COLUMNS = [
    "ts_n_snapshots", "ts_win_first", "ts_win_last",
    "ts_win_min", "ts_win_max", "ts_win_std",
    "ts_win_drop_pct", "ts_win_late_drop_pct",
    "ts_show_lo_last", "ts_show_hi_last",
    "ts_show_first_mid", "ts_show_last_mid", "ts_show_drop_pct",
    "ts_implied_prob_last", "ts_implied_prob_first", "ts_implied_prob_last_norm",
    "ts_pop_rank_change",
    "ts_log_win_votes_last", "ts_log_show_votes_last",
    "ts_anomaly_score",
]

EXTRA_ANABA_BASE_COLUMNS = ["pop", "win_odds"]


def merge_ts_odds_features(
    main_df: pd.DataFrame,
    ts_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """main DataFrame に時系列オッズ集約特徴量を join する。

    対応する時系列オッズが無い行は ts_* = 0、has_ts_odds = 0 で埋める。
    """
    out = main_df.copy()
    out["race_id_ts"] = derive_ts_race_ids_from_main(out)

    if ts_features_df is None or ts_features_df.empty:
        for col in TS_ODDS_FEATURE_COLUMNS:
            out[col] = 0.0
        out["has_ts_odds"] = 0
        out.drop(columns=["race_id_ts"], inplace=True)
        return out

    ts = ts_features_df.copy()
    ts["race_id_ts"] = ts["race_id_ts"].astype("int64")
    ts["horse_num"] = ts["horse_num"].astype(int)

    ts["ts_log_win_votes_last"] = np.log1p(ts["ts_win_votes_last"].astype(float))
    ts["ts_log_show_votes_last"] = np.log1p(ts["ts_show_votes_last"].astype(float))

    ts["ts_anomaly_score"] = (
        ts["ts_win_late_drop_pct"].clip(-3, 3) * 0.6
        + (ts["ts_pop_rank_change"].clip(-10, 10) / 10.0) * 0.4
    )

    merge_cols = ["race_id_ts", "horse_num"] + [
        c for c in TS_ODDS_FEATURE_COLUMNS if c in ts.columns
    ]
    out["horse_num"] = out["horse_num"].astype(int)
    out = out.merge(ts[merge_cols], on=["race_id_ts", "horse_num"], how="left")

    out["has_ts_odds"] = out["ts_n_snapshots"].notna().astype(int)
    for col in TS_ODDS_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0

    out.drop(columns=["race_id_ts"], inplace=True)
    return out


def load_ts_odds_features(
    ts_odds_dir: str | Path,
    years: list[int],
    cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """時系列オッズ集約特徴量をまとめてロード (年単位キャッシュ)。"""
    return load_jd_for_years(
        ts_odds_dir=ts_odds_dir, years=years, cache_dir=cache_dir,
        verbose=verbose, aggregate=True,
    )


def build_target_anaba(df: pd.DataFrame, min_pop: int = 5) -> pd.Series:
    """穴馬ターゲット: ``rank == 1 AND pop >= min_pop`` 二値ラベル。"""
    rank = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    pop = pd.to_numeric(df["pop"], errors="coerce").fillna(0).astype(int)
    return ((rank == 1) & (pop >= min_pop)).astype(int)
