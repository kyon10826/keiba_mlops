"""時系列オッズ特徴量を main DataFrame に結合するユーティリティ。

「市場非依存」原則を保つため、これらの特徴量は基本モデル(複勝予測)には混ぜず、
穴馬モデル専用の追加特徴量として位置付ける。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.time_series_odds import (
    derive_ts_race_ids_from_main,
    load_jd_for_years,
)

# 穴馬モデル専用に追加される時系列オッズ特徴量列
TS_ODDS_FEATURE_COLUMNS = [
    "ts_n_snapshots",
    "ts_win_first",
    "ts_win_last",
    "ts_win_min",
    "ts_win_max",
    "ts_win_std",
    "ts_win_drop_pct",
    "ts_win_late_drop_pct",
    "ts_show_lo_last",
    "ts_show_hi_last",
    "ts_show_first_mid",
    "ts_show_last_mid",
    "ts_show_drop_pct",
    "ts_implied_prob_last",
    "ts_implied_prob_first",
    "ts_implied_prob_last_norm",
    "ts_pop_rank_change",
    "ts_log_win_votes_last",
    "ts_log_show_votes_last",
    "ts_anomaly_score",
]

# 穴馬モデルに追加で渡したい "非時系列" の主要列。
# pop (人気) は穴馬定義 (pop >= min_pop) の核心なのでモデルが直接見られるようにする。
# win_odds は当日確定オッズ。直近の確定値として穴馬判定の根拠になる。
EXTRA_ANABA_BASE_COLUMNS = ["pop", "win_odds"]


def merge_ts_odds_features(
    main_df: pd.DataFrame,
    ts_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """main DataFrame に時系列オッズ集約特徴量を join する。

    Args:
        main_df: 学習用 main DataFrame (race_id, horse_num 列を含む)。
        ts_features_df: ``aggregate_jd_per_horse`` の出力 (race_id_ts, horse_num, ts_* 列)。

    Returns:
        ts_* 列が追加された DataFrame (元のキー列はそのまま)。
        対応する時系列オッズが無い行では ts_* 列は欠損 → 0 / 中央値で埋め (後段の特徴量パイプラインに委譲)。
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

    # 派生特徴量を追加
    ts["ts_log_win_votes_last"] = np.log1p(ts["ts_win_votes_last"].astype(float))
    ts["ts_log_show_votes_last"] = np.log1p(ts["ts_show_votes_last"].astype(float))

    # anomaly_score: オッズ低下率と人気序列上昇の合成 (大きいほど「直前に注目された」)
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
    """学習用に時系列オッズ集約特徴量をまとめてロードする (年単位キャッシュ)。"""
    return load_jd_for_years(
        ts_odds_dir=ts_odds_dir,
        years=years,
        cache_dir=cache_dir,
        verbose=verbose,
        aggregate=True,
    )


def build_target_anaba(df: pd.DataFrame, min_pop: int = 5) -> pd.Series:
    """穴馬ターゲット: ``rank == 1 AND pop >= min_pop`` の二値ラベルを返す。

    pop (人気) が NaN や 0 の行はターゲット 0 (穴馬ではない) とみなす。
    error_code != 0 の行はそもそも学習データから除外されていることを前提とする。
    """
    rank = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    pop = pd.to_numeric(df["pop"], errors="coerce").fillna(0).astype(int)
    return ((rank == 1) & (pop >= min_pop)).astype(int)
