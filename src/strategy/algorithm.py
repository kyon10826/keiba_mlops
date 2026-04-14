"""総合的なレース評価アルゴリズム。

多段階の馬券購入判断パイプラインを実装する:
1. 見送りフィルター
2. 能力評価
3. 展開評価
4. 拮抗時の差別化
5. 割引要因
6. 加点要因
7. オッズ妙味の判定
8. 馬券種別の選定
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 著名なトップ騎手の ID (config から設定可能)
DEFAULT_POPULAR_JOCKEY_IDS = [1088, 5339]  # 川田, ルメール


@dataclass
class HorseEvaluation:
    """1頭の馬の評価結果。"""
    horse_num: int = 0
    horse_name: str = ""
    pred_prob: float = 0.0
    index_rank: int = 0       # 指数順位 (pred_prob 順位)
    ra_rank: int = 0          # RA順位 (移動平均順位)
    ability_score: float = 0.0
    pace_score: float = 0.0
    discount_total: float = 0.0
    plus_total: float = 0.0
    final_score: float = 0.0
    odds_value: float = 0.0   # 能力と市場の乖離度
    win_odds: float = 0.0
    mark: str = ""            # ◎, ○, ▲, △, ""
    discount_reasons: list = field(default_factory=list)
    plus_reasons: list = field(default_factory=list)
    running_style: str = ""   # 逃, 先, 差, 追


@dataclass
class RaceEvaluation:
    """1レース分の評価結果のコンテナ。"""
    race_id: str = ""
    race_name: str = ""
    skip: bool = False
    skip_reason: str = ""
    horses: list = field(default_factory=list)
    ticket_type: str = ""     # tansho_nagashi, umaren_wide, fukusho_wide, box, skip
    ticket_label: str = ""    # 日本語表示
    confidence: str = ""      # ◎, ○○, 穴, 混戦
    dominant_horse: HorseEvaluation | None = None
    recommended_bets: list = field(default_factory=list)


def pre_screen(
    race_df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[bool, str]:
    """見送りフィルター。

    以下の条件に該当するレースをスキップする:
    - 2歳/3歳限定
    - 新馬/未勝利
    - 障害
    - ハンデ
    - 出走頭数 8 頭未満
    - 最大予測確率 0.5 未満

    Returns:
        (skip, reason) のタプル
    """
    if cfg is None:
        cfg = {}
    algo = cfg.get("algorithm", {})
    ps = algo.get("pre_screen", {})

    min_field_size = ps.get("min_field_size", 8)
    min_max_prob = ps.get("min_max_prob", 0.5)

    if race_df.empty:
        return True, "データなし"

    # 出走頭数チェック
    field_size = len(race_df)
    if field_size < min_field_size:
        return True, f"出走頭数不足 ({field_size}頭 < {min_field_size}頭)"

    # レース名に基づくチェック
    race_name = str(race_df["race_name"].iloc[0]) if "race_name" in race_df.columns else ""

    if ps.get("skip_maiden", True):
        if "新馬" in race_name or "未勝利" in race_name:
            return True, f"新馬/未勝利レース: {race_name}"
        # class_code も確認
        cc = race_df["class_code"].iloc[0] if "class_code" in race_df.columns else 0
        if cc == 10:
            return True, "新馬/未勝利クラス"

    if ps.get("skip_obstacle", True):
        if "障害" in race_name or "ジャンプ" in race_name:
            return True, f"障害レース: {race_name}"
        # track_code を確認: 障害は通常 track_code が 3x 番台
        tc = race_df["track_code"].iloc[0] if "track_code" in race_df.columns else 0
        if tc // 10 == 3:
            return True, "障害レース (track_code)"

    if ps.get("skip_handicap", True):
        wc = race_df["weight_code"].iloc[0] if "weight_code" in race_df.columns else 0
        if wc == 3 or "ハンデ" in race_name:
            return True, "ハンデ戦"

    if ps.get("skip_age_limited", True):
        ac = race_df["age_code"].iloc[0] if "age_code" in race_df.columns else 0
        # age_code が 2歳限定 または 3歳限定 の場合
        if ac in (2, 3):
            # 重賞レースかどうかを確認 (上位クラスの3歳戦は例外)
            cc = race_df["class_code"].iloc[0] if "class_code" in race_df.columns else 0
            if cc < 40:  # オープン/重賞ではない
                return True, f"年齢限定レース (age_code={ac})"

    # 最大確率チェック
    if "pred_prob" in race_df.columns:
        max_prob = race_df["pred_prob"].max()
        if max_prob < min_max_prob:
            return True, f"最大確率不足 ({max_prob:.3f} < {min_max_prob})"

    return False, ""


def evaluate_ability(
    race_df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, bool, float]:
    """能力評価 (最重要ステップ)。

    使用指標:
    - 指数順位 = pred_prob 順位 (スピード指数の代替)
    - RA順位 = rank_rolling_3 の順位
    - RPR = pred_prob (キャリブレーション済み複勝確率)
    - ランキングモデル = pred_prob (LightGBM の出力がこの役割を担う)

    Returns:
        (df_with_ability, is_dominant, gap) のタプル
    """
    if cfg is None:
        cfg = {}
    algo = cfg.get("algorithm", {})
    ability_cfg = algo.get("ability", {})
    dominance_rpr = ability_cfg.get("dominance_rpr_threshold", 0.65)
    dominance_gap = ability_cfg.get("dominance_gap_threshold", 0.10)

    df = race_df.copy()

    # 指数順位 = pred_prob の順位 (1 = 最高確率)
    df["index_rank"] = df["pred_prob"].rank(ascending=False, method="min").astype(int)

    # RA順位 = rank_rolling_3 の順位 (1 = 最良の移動平均)
    if "rank_rolling_3" in df.columns:
        # 移動平均値が小さいほど良いので昇順で順位付け
        df["ra_rank"] = df["rank_rolling_3"].rank(ascending=True, method="min").astype(int)
    else:
        df["ra_rank"] = df["index_rank"]

    # 能力スコア = 重み付き合成値
    # 重み: pred_prob 50%, index_rank 25%, ra_rank 25%
    max_horses = len(df)
    df["ability_score"] = (
        df["pred_prob"] * 0.50
        + (1.0 - (df["index_rank"] - 1) / max(max_horses - 1, 1)) * 0.25
        + (1.0 - (df["ra_rank"] - 1) / max(max_horses - 1, 1)) * 0.25
    )

    # 抜けた1頭(ドミナント)であるか確認
    sorted_probs = df["pred_prob"].sort_values(ascending=False)
    top1_rpr = sorted_probs.iloc[0] if len(sorted_probs) > 0 else 0
    top2_rpr = sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0
    gap = top1_rpr - top2_rpr
    is_dominant = (top1_rpr >= dominance_rpr) and (gap >= dominance_gap)

    return df, is_dominant, gap


def estimate_running_style(horse_id: int, hist_df: pd.DataFrame) -> str:
    """過去のコーナー通過順位から脚質を推定する。

    Returns: 逃, 先, 差, 追
    """
    if hist_df is None or hist_df.empty:
        return "不明"

    horse_hist = hist_df[hist_df["id"] == horse_id]
    if horse_hist.empty or "corner4_rank" not in horse_hist.columns:
        return "不明"

    # 直近5レースを使用
    recent = horse_hist.tail(5)
    c4 = recent["corner4_rank"]
    c4 = c4[c4 > 0]  # 無効値を除外

    if c4.empty:
        return "不明"

    field_sizes = recent.loc[c4.index, "horse_N"] if "horse_N" in recent.columns else pd.Series([15] * len(c4))
    field_sizes = field_sizes.replace(0, 15)

    # 相対位置 (0-1, 0=先頭, 1=最後方)
    rel_pos = (c4.values - 1) / (field_sizes.values - 1).clip(min=1)
    avg_rel = np.mean(rel_pos)

    if avg_rel <= 0.15:
        return "逃"
    elif avg_rel <= 0.35:
        return "先"
    elif avg_rel <= 0.65:
        return "差"
    else:
        return "追"


def evaluate_pace(
    race_df: pd.DataFrame,
    hist_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """展開評価。

    各馬の脚質を推定し、先行馬の数をカウントして、
    ペース想定を行い、各馬に展開有利度スコアを付与する。
    """
    df = race_df.copy()
    dist = df["dist"].iloc[0] if "dist" in df.columns else 1600

    # 各馬の脚質を推定
    styles = []
    if hist_df is not None and "id" in df.columns:
        for _, row in df.iterrows():
            style = estimate_running_style(int(row.get("id", 0)), hist_df)
            styles.append(style)
    else:
        styles = ["不明"] * len(df)

    df["running_style"] = styles

    # 先行馬の数をカウント
    n_esc = sum(1 for s in styles if s == "逃")
    n_front = sum(1 for s in styles if s in ("逃", "先"))

    # ペース判定
    # 逃げ馬が多い + 長距離 = ハイペース = 差し/追込馬が有利
    # 逃げ馬が少ない + 短距離 = スローペース = 先行馬が有利
    if n_esc >= 3 or (n_esc >= 2 and dist >= 2000):
        pace = "ハイペース"
    elif n_esc <= 1 and dist <= 1600:
        pace = "スローペース"
    else:
        pace = "ミドルペース"

    # 各馬の展開有利度スコア
    pace_scores = []
    for style in styles:
        if pace == "ハイペース":
            # ハイペースは差し/追込が有利
            score_map = {"逃": -0.10, "先": -0.03, "差": 0.05, "追": 0.08, "不明": 0.0}
        elif pace == "スローペース":
            # スローペースは先行馬が有利
            score_map = {"逃": 0.08, "先": 0.05, "差": -0.03, "追": -0.08, "不明": 0.0}
        else:
            score_map = {"逃": 0.02, "先": 0.02, "差": 0.0, "追": -0.02, "不明": 0.0}

        # 単騎逃げボーナス
        bonus = 0.05 if style == "逃" and n_esc == 1 else 0.0
        pace_scores.append(score_map.get(style, 0.0) + bonus)

    df["pace_score"] = pace_scores
    df["pace_label"] = pace

    return df


def compute_discounts(
    race_df: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """割引要因 (減点) を計算する。

    チェック項目:
    - 昇級初戦
    - 長期休み明け
    - 初距離 (不明としてフラグ)
    - 高齢
    - 気性難 (近似値)
    - 過剰人気
    - 指数なし (データ不足/コールドスタート)
    - 低ランク騎手
    - スローペースでの追い込み不利
    """
    if cfg is None:
        cfg = {}
    algo = cfg.get("algorithm", {})
    dc = algo.get("discount", {})
    popular_jockey_ids = algo.get("popular_jockey_ids", DEFAULT_POPULAR_JOCKEY_IDS)

    long_layoff_days = dc.get("long_layoff_days", 180)
    old_age_threshold = dc.get("old_age_threshold", 8)
    low_jockey_lcb = dc.get("low_jockey_lcb95", 0.15)
    excessive_pop_odds = dc.get("excessive_popularity_odds", 2.0)

    df = race_df.copy()
    discounts = []
    reasons_list = []

    for _, row in df.iterrows():
        discount = 0.0
        reasons = []

        # 長期休み明け
        span = row.get("race_span_days", 0)
        if span and span > long_layoff_days:
            discount += 0.08
            reasons.append(f"長期休み明け({int(span)}日)")

        # 高齢
        age = row.get("age", 0)
        if age and age >= old_age_threshold:
            discount += 0.05
            reasons.append(f"高齢({age}歳)")

        # 過剰人気 - 指数的な裏付けのない人気騎手
        odds = row.get("win_odds", 0)
        jockey_id = row.get("jockey_id", 0)
        pred_prob = row.get("pred_prob", 0)
        index_rank = row.get("index_rank", 99)

        if odds > 0 and odds < excessive_pop_odds:
            # 非常に低いオッズ (大本命)
            if index_rank > 2:
                # 人気だがモデルは支持していない
                discount += 0.10
                reasons.append(f"過剰人気(単{odds:.1f}倍, 指数{index_rank}位)")
            # 特別ルール: 指数の裏付けがないのに極低オッズの人気騎手
            if jockey_id in popular_jockey_ids and odds < 1.5 and index_rank > 1:
                discount += 0.05
                reasons.append("人気騎手過信割引")

        # コールドスタート (移動平均データなし)
        rank_last = row.get("rank_last", 0)
        if rank_last == 0 or pd.isna(rank_last):
            discount += 0.05
            reasons.append("指数なし(初出走/データ不足)")

        # 低ランク騎手
        jockey_lcb = row.get("jockey_lcb95", 0)
        if jockey_lcb and jockey_lcb < low_jockey_lcb:
            discount += 0.04
            reasons.append(f"低ランク騎手(LCB95={jockey_lcb:.3f})")

        # スローペースでの追い込み不利
        pace = row.get("pace_label", "")
        style = row.get("running_style", "")
        if pace == "スローペース" and style == "追":
            discount += 0.06
            reasons.append("スローペースで追い込み不利")

        discounts.append(discount)
        reasons_list.append(reasons)

    df["discount_total"] = discounts
    df["discount_reasons"] = reasons_list

    return df


def compute_plus_factors(
    race_df: pd.DataFrame,
    hist_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """加点要因を計算する。

    チェック項目:
    - 叩き2/3戦目 (休養明けからの2/3戦目)
    - ブリンカー初装着
    - 減量騎手
    - クラス実績
    - 格上騎手乗り替わり
    - 連続騎乗
    - 逃げイチ (単騎逃げ)
    """
    if cfg is None:
        cfg = {}
    algo = cfg.get("algorithm", {})
    pc = algo.get("plus", {})

    df = race_df.copy()
    plus_scores = []
    reasons_list = []

    for _, row in df.iterrows():
        plus = 0.0
        reasons = []

        # ブリンカー装着
        blinker = str(row.get("blinker", "")).strip()
        if blinker and blinker.upper() in ("B", "1"):
            plus += pc.get("blinker_bonus", 0.03)
            reasons.append("ブリンカー装着")

        # 単騎逃げ
        style = row.get("running_style", "")
        if style == "逃" and row.get("pace_label", "") != "ハイペース":
            # pace_score で既に考慮しているが、参照用として印を追加
            n_esc = sum(1 for s in df.get("running_style", []) if s == "逃")
            if n_esc == 1:
                plus += pc.get("lone_frontrunner_bonus", 0.05)
                reasons.append("逃げイチ")

        # 叩き2/3戦目 (休養明けからの2/3戦目)
        # 近似: 現在の race_span_days は短いが前走は長期間あいていた
        span = row.get("race_span_days", 0)
        momentum = row.get("label_momentum", 0)
        if span and 14 <= span <= 60 and momentum > 0:
            plus += 0.04
            reasons.append("叩き2-3戦目(上昇)")

        # 騎手 LCB95 が高い (格上騎手)
        jockey_lcb = row.get("jockey_lcb95", 0)
        if jockey_lcb and jockey_lcb >= 0.30:
            plus += pc.get("top_jockey_bonus", 0.03)
            reasons.append(f"上位騎手(LCB95={jockey_lcb:.3f})")

        # 良好なクラス実績 (show_rate_last_5 > 0.5)
        show_rate = row.get("show_rate_last_5", 0)
        if show_rate and show_rate > 0.5:
            plus += 0.03
            reasons.append(f"クラス好実績(複勝率{show_rate:.0%})")

        # 減量騎手: basis_weight が明確に低い
        bw = row.get("basis_weight", 0)
        if bw and bw > 0 and bw < 54:
            plus += 0.02
            reasons.append(f"減量騎手({bw}kg)")

        plus_scores.append(plus)
        reasons_list.append(reasons)

    df["plus_total"] = plus_scores
    df["plus_reasons"] = reasons_list

    return df


def compute_odds_value(
    race_df: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """オッズ妙味の判定。

    能力から導かれる適正オッズと実際の市場オッズを比較する。
    正の値 = 市場に過小評価されている馬。
    """
    df = race_df.copy()

    if "win_odds" not in df.columns or (df["win_odds"] <= 0).all():
        df["odds_value"] = 0.0
        df["odds_label"] = "---"
        return df

    # 能力スコアから導かれる適正複勝オッズ
    # 複勝の適正オッズ ≈ 1 / (ability_score * 3)
    # オッズ妙味: pred_prob から導かれるインプライドと市場オッズを比較
    odds_values = []
    labels = []

    for _, row in df.iterrows():
        prob = row.get("pred_prob", 0)
        odds = row.get("win_odds", 0)

        if prob <= 0 or odds <= 0:
            odds_values.append(0.0)
            labels.append("---")
            continue

        # モデルから導かれる適正単勝オッズ
        # 簡易変換: show_prob = pred_prob のとき win_prob ≈ pred_prob / 3
        # 適正オッズ ≈ 1 / win_prob_est
        win_prob_est = max(prob / 3.0, 0.01)
        fair_odds = 1.0 / win_prob_est

        # 妙味 = (実オッズ / 適正オッズ) - 1.0
        # 正 = 市場が過小評価、負 = 市場が過大評価
        value = (odds / fair_odds) - 1.0
        odds_values.append(value)

        if value >= 0.5:
            labels.append("割安")
        elif value >= 0.0:
            labels.append("適正")
        elif value >= -0.3:
            labels.append("やや割高")
        else:
            labels.append("割高")

    df["odds_value"] = odds_values
    df["odds_label"] = labels

    return df


def assign_marks(
    race_df: pd.DataFrame,
    is_dominant: bool,
) -> pd.DataFrame:
    """総合スコアに基づいて印 (◎○▲△) を付与する。"""
    df = race_df.copy()
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    marks = [""] * len(df)
    if len(df) >= 1:
        if is_dominant:
            marks[0] = "◎"
        else:
            marks[0] = "○"
    if len(df) >= 2:
        if is_dominant:
            marks[1] = "○"
        else:
            marks[1] = "○"
    if len(df) >= 3:
        marks[2] = "▲"
    if len(df) >= 4:
        marks[3] = "△"
    if len(df) >= 5:
        marks[4] = "△"

    df["mark"] = marks
    return df


def determine_ticket_type(
    evaluation: RaceEvaluation,
    is_dominant: bool,
    gap: float,
    cfg: dict | None = None,
) -> RaceEvaluation:
    """評価結果に基づいて馬券種別を決定する。

    ◎ 1頭軸 → 単勝 + 馬連流し
    ○○ 拮抗 → 馬連・ワイド
    穴狙い  → 複勝・ワイド
    混戦    → ボックス または 見送り
    """
    horses = evaluation.horses
    if not horses:
        evaluation.ticket_type = "skip"
        evaluation.ticket_label = "見送り"
        evaluation.confidence = "---"
        return evaluation

    sorted_h = sorted(horses, key=lambda h: h.final_score, reverse=True)
    top = sorted_h[0]

    if is_dominant and top.pred_prob >= 0.65:
        # 明確な抜けた1頭
        if top.odds_value >= 0.0:
            evaluation.ticket_type = "tansho_nagashi"
            evaluation.ticket_label = "◎ 単勝+馬連流し"
            evaluation.confidence = "◎"
        else:
            evaluation.ticket_type = "umaren_wide"
            evaluation.ticket_label = "◎ 馬連・ワイド(割高注意)"
            evaluation.confidence = "◎"
        evaluation.dominant_horse = top

    elif len(sorted_h) >= 2 and (sorted_h[0].final_score - sorted_h[1].final_score) < 0.05:
        # 拮抗
        if top.odds_value >= 0.5:
            # 穴馬妙味
            evaluation.ticket_type = "fukusho_wide"
            evaluation.ticket_label = "穴狙い 複勝・ワイド"
            evaluation.confidence = "穴"
        else:
            evaluation.ticket_type = "umaren_wide"
            evaluation.ticket_label = "○○ 馬連・ワイド"
            evaluation.confidence = "○○"

    elif top.final_score >= 0.40:
        # 中程度の信頼度
        if top.odds_value >= 0.3:
            evaluation.ticket_type = "fukusho_wide"
            evaluation.ticket_label = "穴狙い 複勝・ワイド"
            evaluation.confidence = "穴"
        else:
            evaluation.ticket_type = "tansho_nagashi"
            evaluation.ticket_label = "◎ 単勝+馬連流し"
            evaluation.confidence = "◎"
        evaluation.dominant_horse = top

    else:
        # 混戦 または 低信頼度
        if len(sorted_h) >= 3 and sorted_h[2].final_score >= 0.30:
            evaluation.ticket_type = "box"
            evaluation.ticket_label = "混戦 ボックス"
            evaluation.confidence = "混戦"
        else:
            evaluation.ticket_type = "skip"
            evaluation.ticket_label = "見送り"
            evaluation.confidence = "---"

    return evaluation


def build_bet_recommendations(
    evaluation: RaceEvaluation,
    cfg: dict | None = None,
) -> RaceEvaluation:
    """馬券種別に応じて具体的な購入推奨を構築する。"""
    if cfg is None:
        cfg = {}
    strat = cfg.get("strategy", {})

    horses = sorted(evaluation.horses, key=lambda h: h.final_score, reverse=True)
    bets = []

    if evaluation.ticket_type == "tansho_nagashi" and horses:
        # ◎に単勝、◎→○▲△に馬連流し
        axis = horses[0]
        bets.append(f"単勝: {axis.horse_num}番 {axis.horse_name}")
        targets = [h for h in horses[1:4] if h.mark in ("○", "▲")]
        if targets:
            target_nums = [str(h.horse_num) for h in targets]
            bets.append(f"馬連流し: {axis.horse_num}→{','.join(target_nums)}")

    elif evaluation.ticket_type == "umaren_wide" and len(horses) >= 2:
        # 上位2-3頭で馬連・ワイド
        top = horses[:3]
        nums = [str(h.horse_num) for h in top]
        bets.append(f"馬連: {'-'.join(nums[:2])}")
        if len(nums) >= 2:
            bets.append(f"ワイド: {'-'.join(nums[:2])}")
        if len(nums) >= 3:
            bets.append(f"ワイド: {nums[0]}-{nums[2]}")

    elif evaluation.ticket_type == "fukusho_wide" and horses:
        # 妙味のある馬に複勝・ワイド
        value_horses = [h for h in horses[:3] if h.odds_value >= 0.0]
        if not value_horses:
            value_horses = horses[:2]
        for h in value_horses:
            bets.append(f"複勝: {h.horse_num}番 {h.horse_name}")
        if len(value_horses) >= 2:
            bets.append(f"ワイド: {value_horses[0].horse_num}-{value_horses[1].horse_num}")

    elif evaluation.ticket_type == "box" and len(horses) >= 3:
        top3 = horses[:3]
        nums = [str(h.horse_num) for h in top3]
        bets.append(f"三連複BOX: {'-'.join(nums)}")
        bets.append(f"ワイドBOX: {'-'.join(nums)}")

    evaluation.recommended_bets = bets
    return evaluation


def run_full_evaluation(
    race_df: pd.DataFrame,
    hist_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> RaceEvaluation:
    """1レースに対して多段階の評価パイプライン全体を実行する。

    Args:
        race_df: 予測結果を含む1レース分の DataFrame
        hist_df: 脚質推定用の過去データ
        cfg: 設定辞書全体

    Returns:
        完全な分析結果を含む RaceEvaluation
    """
    if cfg is None:
        cfg = {}

    eval_result = RaceEvaluation()
    eval_result.race_name = str(race_df["race_name"].iloc[0]) if "race_name" in race_df.columns else ""

    # ステップ1: 見送りフィルター
    skip, reason = pre_screen(race_df, cfg)
    if skip:
        eval_result.skip = True
        eval_result.skip_reason = reason
        eval_result.ticket_type = "skip"
        eval_result.ticket_label = f"見送り: {reason}"
        return eval_result

    # ステップ2: 能力評価
    df, is_dominant, gap = evaluate_ability(race_df, cfg)

    # ステップ3: 展開評価
    df = evaluate_pace(df, hist_df)

    # ステップ4: 割引要因
    df = compute_discounts(df, cfg)

    # ステップ5: 加点要因
    df = compute_plus_factors(df, hist_df, cfg)

    # ステップ6: オッズ妙味
    df = compute_odds_value(df, cfg)

    # 総合スコアを計算
    df["final_score"] = (
        df["ability_score"]
        + df["pace_score"]
        - df["discount_total"]
        + df["plus_total"]
    )

    # 特別ルール: 能力が明確 (RPR>=0.65 かつギャップあり) なら展開不利でも評価を維持
    if is_dominant:
        top_idx = df["pred_prob"].idxmax()
        if df.loc[top_idx, "pace_score"] < 0:
            # 抜けた1頭の展開ペナルティを無効化
            df.loc[top_idx, "final_score"] = (
                df.loc[top_idx, "ability_score"]
                - df.loc[top_idx, "discount_total"]
                + df.loc[top_idx, "plus_total"]
            )

    # 印を付与
    df = assign_marks(df, is_dominant)

    # 各馬の評価結果を構築
    horse_evals = []
    for _, row in df.iterrows():
        he = HorseEvaluation(
            horse_num=int(row.get("horse_num", 0)),
            horse_name=str(row.get("horse", "")),
            pred_prob=float(row.get("pred_prob", 0)),
            index_rank=int(row.get("index_rank", 0)),
            ra_rank=int(row.get("ra_rank", 0)),
            ability_score=float(row.get("ability_score", 0)),
            pace_score=float(row.get("pace_score", 0)),
            discount_total=float(row.get("discount_total", 0)),
            plus_total=float(row.get("plus_total", 0)),
            final_score=float(row.get("final_score", 0)),
            odds_value=float(row.get("odds_value", 0)),
            win_odds=float(row.get("win_odds", 0)),
            mark=str(row.get("mark", "")),
            discount_reasons=row.get("discount_reasons", []),
            plus_reasons=row.get("plus_reasons", []),
            running_style=str(row.get("running_style", "")),
        )
        horse_evals.append(he)

    eval_result.horses = horse_evals

    # ステップ7-8: 馬券種別と推奨購入
    eval_result = determine_ticket_type(eval_result, is_dominant, gap, cfg)
    eval_result = build_bet_recommendations(eval_result, cfg)

    return eval_result


def print_evaluation(evaluation: RaceEvaluation) -> None:
    """評価結果全体を整形して出力する。"""
    print("\n" + "=" * 70)
    print("  総合評価アルゴリズム結果")
    print("=" * 70)

    if evaluation.race_name:
        print(f"  レース: {evaluation.race_name}")

    if evaluation.skip:
        print(f"\n  >>> 見送り: {evaluation.skip_reason}")
        print("=" * 70)
        return

    # ペース情報
    if evaluation.horses:
        pace = ""
        styles_count = {}
        for h in evaluation.horses:
            s = h.running_style
            styles_count[s] = styles_count.get(s, 0) + 1
        style_str = ", ".join(f"{k}:{v}" for k, v in sorted(styles_count.items()) if k != "不明")
        print(f"  脚質分布: {style_str}")

    print(f"\n  判定: {evaluation.confidence}  馬券: {evaluation.ticket_label}")

    # 馬一覧テーブル
    print(f"\n  {'印':>2s}  {'馬番':>4s}  {'馬名':<12s}  {'RPR':>5s}  "
          f"{'能力':>5s}  {'展開':>5s}  {'割引':>5s}  {'加点':>5s}  "
          f"{'総合':>5s}  {'オッズ':>6s}  {'妙味':>6s}  {'脚質':>4s}")
    print("  " + "-" * 88)

    for h in sorted(evaluation.horses, key=lambda x: x.final_score, reverse=True):
        mark = h.mark if h.mark else "  "
        odds_str = f"{h.win_odds:.1f}" if h.win_odds > 0 else "---"
        value_str = f"{h.odds_value:+.2f}" if h.odds_value != 0 else "---"
        print(f"  {mark:>2s}  {h.horse_num:>4d}  {h.horse_name:<12s}  "
              f"{h.pred_prob:.3f}  {h.ability_score:.3f}  {h.pace_score:+.3f}  "
              f"{-h.discount_total:+.3f}  {h.plus_total:+.3f}  "
              f"{h.final_score:.3f}  {odds_str:>6s}  {value_str:>6s}  {h.running_style:>4s}")

    # 印を付けた馬の割引/加点詳細
    marked = [h for h in evaluation.horses if h.mark in ("◎", "○", "▲")]
    if marked:
        print(f"\n  --- 注目馬の詳細 ---")
        for h in marked:
            print(f"\n  {h.mark} {h.horse_num}番 {h.horse_name}")
            if h.discount_reasons:
                print(f"    割引: {', '.join(h.discount_reasons)}")
            if h.plus_reasons:
                print(f"    加点: {', '.join(h.plus_reasons)}")

    # 推奨馬券
    if evaluation.recommended_bets:
        print(f"\n  --- 推奨馬券 ---")
        for bet in evaluation.recommended_bets:
            print(f"    {bet}")

    print("\n" + "=" * 70)
