"""特徴量パイプラインのオーケストレーション。

================================================================================
【レビューガイド】このファイルの役割と設計判断
================================================================================
- 生の record_data (47 列/行 = 1出走) を、モデル入力の 46 特徴量に変換する。
- sklearn の Pipeline と同じ「fit → transform」パターン:
    * fit(train_df):   学習データ**だけ**から統計量 (騎手/種牡馬のベイズ事後平均、
                       コールドスタート補完値) を算出して保持する。
                       → valid/test の情報を使わないことでリークを防ぐ。
    * transform(df):   fit 済み統計量を使って任意のデータを特徴量化する。
                       学習時は train+valid+test を「縦結合してから」一度に transform する。
                       理由: ローリング特徴量 (直近3走平均など) は年をまたぐため、
                       年別に処理すると各年の先頭で履歴が切れてしまう。
- 5 つの特徴量モジュールを順に適用する:
    race (レース条件) → horse (馬のローリング) → jockey (騎手) → sire (血統)
    → relative (レース内相対値)

【重要な不変条件】transform は入力と同じ行順で返す (詳細は transform の docstring)。
呼び出し側 (scripts/train.py 等) は変換後を iloc[:n_train] で再分割するため、
この保証が崩れると train/test 分割そのものが壊れる (過去に実際に壊れていた)。
================================================================================
"""

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

# 学習に使用する特徴量列の順序付きリスト (計 46 列)
#
# ★ オッズ不使用方針: 市場情報 (win_odds, pop, odds_rank) は入力しない。
#   理由 (2つ):
#   1. 差別化 — オッズを入れると予測が市場の合意をなぞるだけになり独自性が消える
#   2. 当日運用の頑健性 — オッズ入力に依存すると 5 分前オッズ API 障害時に予測不能になる
#   ※ ただし「投票額の決定」ではオッズを使う (EV 計算・オッズ帯判定)。
#     「モデル入力に使わない」と「投票判断に使わない」は別、という切り分け。
#
# 列の由来はコメントのグループ分け参照。各グループの生成ロジック:
#   レース      → src/features/race.py     (レース条件のエンコード)
#   ローリング  → src/features/horse.py    (馬ごとの過去成績。全て shift(1) でリーク防止)
#   騎手/種牡馬 → src/features/jockey.py, sire.py (Beta-Binomial 事後平均 + 95%信頼下限)
#   相対        → src/features/relative.py (同レース出走馬内での z-score 等)
FEATURE_COLUMNS = [
    # レース
    "place_encoded", "track_type", "dist", "dist_category",
    "condition_encoded", "weather_encoded", "class_grade", "field_size",
    # 馬のローリング (基本)
    "rank_last", "rank_rolling_3", "rank_rolling_5", "show_rate_last_5",
    "last_3f_rolling_3", "time_diff_rolling_3",
    "weight_horse", "weight_change", "race_span_days",
    "prize_cumsum", "label_momentum",
    # 業界知見 (Phase 2 で追加)
    "dist_change", "venue_change", "class_change", "blinker_first",
    "win_rate_last_3m", "show_rate_last_3m",
    "last_3f_rank_rolling_3", "corner4_rank_last", "pci_last",
    # 騎手
    "jockey_encoded", "jockey_show_rate", "jockey_win_rate",
    "jockey_race_count", "jockey_lcb95",
    # 種牡馬
    "father_encoded", "sire_show_rate", "sire_lcb95",
    "sire_show_rate_turf", "sire_show_rate_dirt",
    # 相対 (オッズ非依存)
    "weight_zscore", "age_relative",
    "basis_weight_zscore", "prize_zscore", "last_3f_relative",
    # そのまま通す生値
    "horse_num", "waku_num", "age",
]

# LightGBM にカテゴリ変数として渡す列。
# LightGBM は整数エンコードされたカテゴリを「順序なし」として最適分割できる
# (one-hot 不要)。特に jockey/father は数千カテゴリあるため native categorical が有効。
CATEGORICAL_FEATURES = [
    "place_encoded", "track_type", "dist_category",
    "condition_encoded", "weather_encoded",
    "father_encoded", "jockey_encoded",
]

# 市場情報系 (オッズ / 人気) の列名。オッズ不使用アサートで使用する。
MARKET_INFO_COLUMNS = {
    "win_odds", "pop", "odds_rank",
    "show_odds_min", "show_odds_max", "show_odds_avg",
}


def assert_no_market_info(feature_columns: list[str]) -> None:
    """特徴量リストに市場情報が混ざっていないことを保証する。

    AI 競馬予想マスターズのオッズ不使用方針を強制するためのランタイムチェック。
    ts_ (時系列オッズ) 系も除外する。
    """
    leaked = [c for c in feature_columns if c in MARKET_INFO_COLUMNS or c.startswith("ts_")]
    if leaked:
        raise ValueError(
            f"[market-info leak] 以下の列が特徴量に混入している: {leaked}. "
            "AI 競馬予想マスターズはオッズ/人気を特徴量に入れない方針。"
        )


def build_target(df: pd.DataFrame) -> pd.Series:
    """二値ターゲットを作成する: 複勝 (1〜3 着) = 1。

    陽性率 ~21% とクラスバランスが良く、統計的に安定して学習できる。
    rank >= 1 の条件は rank=0 (未入線/データ欠損) を陰性扱いにするため。
    """
    return ((df["rank"] >= 1) & (df["rank"] <= 3)).astype(int)


def build_target_win(df: pd.DataFrame) -> pd.Series:
    """二値ターゲットを作成する: 単勝 (1 着) = 1。

    AI 競馬予想マスターズの主戦場である単勝予測用。陽性率 ~7%
    (平均 14 頭立て → 1/14)。複勝より不均衡なため、学習側では
    scale_pos_weight で補正する (src/model/win_trainer.py 参照)。
    """
    return (df["rank"] == 1).astype(int)


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
        """学習データから統計量を算出する。

        ★ リーク防止の要: ここに渡すのは train_df だけ。
          騎手/種牡馬の成績率を valid/test 期間まで含めて計算すると、
          「未来の成績を知っている特徴量」になってしまう (target leakage の一種)。
        """
        # 騎手/種牡馬の複勝率・勝率を Beta-Binomial 事後平均で平滑化して算出。
        # 出走数が少ない騎手は事前分布 Beta(α=2, β=5) ≒ 複勝率 28.6% に引っ張られ、
        # 「3 戦 3 複勝の新人 = 複勝率 100%」のような過大評価を防ぐ。
        self.jockey_stats = compute_jockey_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        self.sire_stats = compute_sire_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        # コールドスタート補完値: 初出走馬 (過去成績ゼロ) のローリング特徴量を
        # 「同じ父を持つ馬たちの平均値」で埋めるための辞書。
        # 血統は初出走馬に対する数少ない事前情報なので、全体平均より情報量が多い。
        train_with_rolling = add_horse_features(train_df)
        self.cold_start_defaults = compute_cold_start_defaults(train_with_rolling)

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """すべての特徴量変換を適用する。

        学習データの場合、まず全年を結合して馬のローリング特徴量を算出し、
        その後で元に分割する。推論データの場合は履歴データと結合する。

        ★ 出力は **入力と同じ行順** で返すことを保証する。
           (add_horse_features が内部で (id, race_id) ソートするため、
            _orig_order 列で元の順序に復元する。呼び出し側の iloc ベースの
            train/valid/test 再分割はこの保証に依存している。)

        Args:
            df: 入力 DataFrame。
            is_train: True の場合、学習データとして扱い統計量を再計算する。
        """
        # レース特徴量 (場・馬場・天候などのエンコード)
        out, self.place_map, self.weather_map = add_race_features(df)

        # 入力順を記録 (add_horse_features のソートから復元するため)
        # 【バグ修正の経緯】add_horse_features は内部で (id, race_id) ソートしたまま
        # 返すため、以前はここで行順が壊れていた。馬 id は生年で始まるので、
        # iloc[:n_train] の再分割が「時系列分割」ではなく「馬の生年コホート分割」に
        # 化けてしまい、バックテストの母集団が偏っていた。
        # _orig_order を打って最後に復元することで、この不変条件を保証する。
        out = out.reset_index(drop=True)
        out["_orig_order"] = range(len(out))

        # 馬のローリング特徴量 (内部で id, race_id ソートされる。
        # 馬ごとに時系列順へ並べないと「直近 3 走」の rolling が計算できないため)
        out = add_horse_features(out, cold_start_defaults=self.cold_start_defaults)

        # 騎手特徴量 (fit 済み jockey_stats を jockey_id で left join)
        out, self.jockey_map = add_jockey_features(out, self.jockey_stats)

        # 種牡馬特徴量 (fit 済み sire_stats を father で left join)
        out, self.father_map = add_sire_features(out, self.sire_stats)

        # 相対特徴量 (同一レース内の z-score 等。groupby(race) で計算)
        out = add_relative_features(out)

        # 入力順に復元 (不変条件: 入力と同じ行順・同じ行数で返す)
        out = out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)

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
