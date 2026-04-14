"""特徴量パイプラインのオーケストレーション。"""

from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from src.features.race import add_race_features
from src.features.horse import add_horse_features, compute_cold_start_defaults
from src.features.jockey import compute_jockey_stats, add_jockey_features
from src.features.sire import compute_sire_stats, add_sire_features
from src.features.relative import add_relative_features

# 学習に使用する特徴量列の順序付きリスト
FEATURE_COLUMNS = [
    # レース
    "place_encoded", "track_type", "dist", "dist_category",
    "condition_encoded", "weather_encoded", "class_grade", "field_size",
    # 馬のローリング
    "rank_last", "rank_rolling_3", "rank_rolling_5", "show_rate_last_5",
    "last_3f_rolling_3", "time_diff_rolling_3",
    "weight_horse", "weight_change", "race_span_days",
    "prize_cumsum", "label_momentum",
    # 騎手
    "jockey_encoded", "jockey_show_rate", "jockey_win_rate",
    "jockey_race_count", "jockey_lcb95",
    # 種牡馬
    "father_encoded", "sire_show_rate", "sire_lcb95",
    "sire_show_rate_turf", "sire_show_rate_dirt",
    # 相対
    "odds_rank", "weight_zscore", "age_relative",
    # そのまま通す生値
    "horse_num", "waku_num", "age",
]

CATEGORICAL_FEATURES = [
    "place_encoded", "track_type", "dist_category",
    "condition_encoded", "weather_encoded",
    "father_encoded", "jockey_encoded",
]


def build_target(df: pd.DataFrame) -> pd.Series:
    """二値ターゲットを作成する: 複勝 (1〜3 着) = 1。"""
    return ((df["rank"] >= 1) & (df["rank"] <= 3)).astype(int)


class FeaturePipeline:
    """全モジュールにわたる特徴量生成をオーケストレーションする。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        bayesian = cfg.get("bayesian", {})
        self.alpha_prior = bayesian.get("alpha_prior", 2)
        self.beta_prior = bayesian.get("beta_prior", 5)

        # fit 時に保存される
        self.jockey_stats: pd.DataFrame | None = None
        self.sire_stats: pd.DataFrame | None = None
        self.cold_start_defaults: dict | None = None
        self.place_map: dict | None = None
        self.weather_map: dict | None = None
        self.jockey_map: dict | None = None
        self.father_map: dict | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """学習データから統計量を算出する。"""
        self.jockey_stats = compute_jockey_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        self.sire_stats = compute_sire_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        # コールドスタートのデフォルト値を算出 (ローリング特徴量が必要)
        train_with_rolling = add_horse_features(train_df)
        self.cold_start_defaults = compute_cold_start_defaults(train_with_rolling)

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """すべての特徴量変換を適用する。

        学習データの場合、まず全年を結合して馬のローリング特徴量を算出し、
        その後で元に分割する。推論データの場合は履歴データと結合する。

        Args:
            df: 入力 DataFrame。
            is_train: True の場合、学習データとして扱い統計量を再計算する。
        """
        # レース特徴量
        out, self.place_map, self.weather_map = add_race_features(df)

        # 馬のローリング特徴量 (id, race_id でソート済みである必要がある)
        out = add_horse_features(out, cold_start_defaults=self.cold_start_defaults)

        # 騎手特徴量
        out, self.jockey_map = add_jockey_features(out, self.jockey_stats)

        # 種牡馬特徴量
        out, self.father_map = add_sire_features(out, self.sire_stats)

        # 相対特徴量
        out = add_relative_features(out)

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """学習データに対して fit し、同時に変換する。"""
        self.fit(train_df)
        return self.transform(train_df, is_train=True)

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """変換済み DataFrame から特徴量列を抽出する。"""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].copy()

    def save(self, path: str) -> None:
        """パイプラインの状態 (統計量とマッピング) を永続化する。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "jockey_stats": self.jockey_stats,
            "sire_stats": self.sire_stats,
            "cold_start_defaults": self.cold_start_defaults,
            "place_map": self.place_map,
            "weather_map": self.weather_map,
            "jockey_map": self.jockey_map,
            "father_map": self.father_map,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, cfg: dict | None = None) -> "FeaturePipeline":
        """保存済みパイプラインを読み込む。"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        pipe = cls(cfg or {})
        pipe.jockey_stats = state["jockey_stats"]
        pipe.sire_stats = state["sire_stats"]
        pipe.cold_start_defaults = state.get("cold_start_defaults", None)
        pipe.place_map = state["place_map"]
        pipe.weather_map = state["weather_map"]
        pipe.jockey_map = state["jockey_map"]
        pipe.father_map = state["father_map"]
        pipe.alpha_prior = state["alpha_prior"]
        pipe.beta_prior = state["beta_prior"]
        return pipe
