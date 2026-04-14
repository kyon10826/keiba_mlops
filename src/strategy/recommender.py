"""統合的な馬券推奨モジュール。

モデルの予測、オッズデータ、Harville確率、ケリー基準を組み合わせて、
複勝・単勝・三連複・三連単の推奨を生成する。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategy.harville import (
    generate_trifecta_combinations,
    generate_trio_combinations,
    win_probability_from_show,
)
from src.strategy.kelly import compute_bet_amount, compute_tier_bet_amount, compute_bet_amount_dispatch

logger = logging.getLogger(__name__)


def _get_tier_label(prob: float, low: float = 0.3, mid: float = 0.4, high: float = 0.5) -> str:
    """予測確率に基づいてティアラベルを返す。"""
    if prob >= high:
        return "Buy Aggressive"
    elif prob >= mid:
        return "Buy"
    elif prob >= low:
        return "Buy Low"
    return "No Bet"


def recommend_show(
    race_feat: pd.DataFrame,
    min_ev: float = 1.0,
    bankroll: float = 1_000_000,
    kelly_frac: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
    method: str = "threshold",
    prob_threshold: float = 0.3,
    **tier_kwargs,
) -> pd.DataFrame:
    """期待値方式または閾値方式で複勝を推奨する。

    Args:
        race_feat: 次の列を持つ DataFrame: horse_num, horse (馬名),
            pred_prob, show_odds_min, show_odds_max。
        min_ev: 最小期待値閾値(method="ev" 時に使用)。
        bankroll: ケリーサイジングに用いる現在のバンクロール(method="ev" 時)。
        kelly_frac: ケリー倍率(method="ev" 時)。
        max_bet_fraction: バンクロール比の最大賭け金(method="ev" 時)。
        min_bet: 最小賭け金(method="ev" 時)。
        method: "threshold"(閾値方式)または "ev"(従来方式)。
        prob_threshold: 閾値方式での最小 pred_prob。
        **tier_kwargs: compute_tier_bet_amount に渡すティアサイジングパラメータ。

    Returns:
        pred_prob(threshold)または期待値(ev)の降順でソートされた DataFrame。
    """
    if race_feat.empty:
        return pd.DataFrame()

    df = race_feat.copy()

    if method == "threshold":
        # 確率閾値でフィルタリング
        eligible = df[df["pred_prob"] >= prob_threshold].copy()
        if eligible.empty:
            return pd.DataFrame()

        # ティアベースの賭け金サイジング
        tier_kw = {k: v for k, v in tier_kwargs.items() if k.startswith("tier_")}
        eligible["tier"] = eligible["pred_prob"].apply(
            lambda p: _get_tier_label(
                p,
                low=tier_kw.get("tier_low_threshold", 0.3),
                mid=tier_kw.get("tier_mid_threshold", 0.4),
                high=tier_kw.get("tier_high_threshold", 0.5),
            )
        )
        eligible["bet_amount"] = eligible["pred_prob"].apply(
            lambda p: compute_tier_bet_amount(p, **tier_kw)
        )

        # 出力列を構築
        out_cols = ["horse_num"]
        if "horse" in eligible.columns:
            out_cols.append("horse")
        out_cols.extend(["pred_prob", "tier", "bet_amount"])
        # 参照用として show_odds_avg/ev が利用可能なら含める
        if "show_odds_min" in eligible.columns and "show_odds_max" in eligible.columns:
            eligible["show_odds_avg"] = (eligible["show_odds_min"] + eligible["show_odds_max"]) / 2.0
            eligible["show_odds_avg"] = eligible["show_odds_avg"].apply(lambda x: np.nan if x <= 0 else x)
            eligible["ev"] = eligible["pred_prob"] * eligible["show_odds_avg"]
            out_cols.extend(["show_odds_avg", "ev"])
        elif "show_odds_avg" in eligible.columns:
            eligible["ev"] = eligible["pred_prob"] * eligible["show_odds_avg"]
            out_cols.extend(["show_odds_avg", "ev"])

        result = eligible[out_cols].sort_values("pred_prob", ascending=False).reset_index(drop=True)
        return result

    # method == "ev": 従来の期待値ベースのアプローチ
    cols = ["horse_num", "pred_prob", "show_odds_avg", "ev", "bet_amount"]

    # 平均複勝オッズを計算
    if "show_odds_min" in df.columns and "show_odds_max" in df.columns:
        df["show_odds_avg"] = (df["show_odds_min"] + df["show_odds_max"]) / 2.0
    elif "show_odds_avg" in df.columns:
        pass  # 既に存在
    else:
        logger.warning("No show odds columns found, cannot compute show EV")
        return pd.DataFrame(columns=["horse_num", "horse"] + cols[1:])

    # ゼロ/負/不正なオッズを置換 (netkeiba は取消時に -3.0 を返す)
    df["show_odds_avg"] = df["show_odds_avg"].apply(
        lambda x: np.nan if x <= 0 else x
    )

    # 期待値 = pred_prob * show_odds_avg
    df["ev"] = df["pred_prob"] * df["show_odds_avg"]

    # 最小期待値でフィルタリング
    eligible = df[df["ev"] >= min_ev].copy()

    if eligible.empty:
        return pd.DataFrame(columns=["horse_num", "horse"] + cols[1:])

    # ケリー方式による賭け金サイジング
    eligible["bet_amount"] = eligible.apply(
        lambda row: compute_bet_amount(
            prob=row["pred_prob"],
            odds=row["show_odds_avg"],
            bankroll=bankroll,
            fraction=kelly_frac,
            max_bet_fraction=max_bet_fraction,
            min_bet=min_bet,
        )
        if pd.notna(row["show_odds_avg"]) and row["show_odds_avg"] > 0
        else 0.0,
        axis=1,
    )

    # 出力列を選択
    out_cols = ["horse_num"]
    if "horse" in eligible.columns:
        out_cols.append("horse")
    out_cols.extend(["pred_prob", "show_odds_avg", "ev", "bet_amount"])

    result = eligible[out_cols].sort_values("ev", ascending=False).reset_index(drop=True)
    return result


def recommend_win(
    race_feat: pd.DataFrame,
    min_ev: float = 1.0,
    bankroll: float = 1_000_000,
    kelly_frac: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
    method: str = "threshold",
    prob_threshold: float = 0.3,
    **tier_kwargs,
) -> pd.DataFrame:
    """期待値方式または閾値方式で単勝を推奨する。

    Args:
        race_feat: 次の列を持つ DataFrame: horse_num, horse (馬名),
            pred_prob, win_odds。
        min_ev: 最小期待値閾値(method="ev" 時に使用)。
        bankroll: ケリーサイジングに用いる現在のバンクロール(method="ev" 時)。
        kelly_frac: ケリー倍率(method="ev" 時)。
        max_bet_fraction: バンクロール比の最大賭け金(method="ev" 時)。
        min_bet: 最小賭け金(method="ev" 時)。
        method: "threshold"(閾値方式)または "ev"(従来方式)。
        prob_threshold: 閾値方式での最小 pred_prob。
        **tier_kwargs: compute_tier_bet_amount に渡すティアサイジングパラメータ。

    Returns:
        pred_prob(threshold)または期待値(ev)の降順でソートされた DataFrame。
    """
    if race_feat.empty:
        return pd.DataFrame()

    df = race_feat.copy()

    # win_odds がある場合、取消/無効の馬を除外
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # 複勝 pred_prob から単勝確率を推定 (Harville)
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)
    df["win_prob"] = win_probs

    if method == "threshold":
        # pred_prob の確率閾値でフィルタリング
        eligible = df[df["pred_prob"] >= prob_threshold].copy()
        if eligible.empty:
            return pd.DataFrame()

        # ティアベースの賭け金サイジング
        tier_kw = {k: v for k, v in tier_kwargs.items() if k.startswith("tier_")}
        eligible["tier"] = eligible["pred_prob"].apply(
            lambda p: _get_tier_label(
                p,
                low=tier_kw.get("tier_low_threshold", 0.3),
                mid=tier_kw.get("tier_mid_threshold", 0.4),
                high=tier_kw.get("tier_high_threshold", 0.5),
            )
        )
        eligible["bet_amount"] = eligible["pred_prob"].apply(
            lambda p: compute_tier_bet_amount(p, **tier_kw)
        )

        out_cols = ["horse_num"]
        if "horse" in eligible.columns:
            out_cols.append("horse")
        out_cols.extend(["pred_prob", "win_prob", "tier", "bet_amount"])
        # 参照用として win_odds/ev が利用可能なら含める
        if "win_odds" in eligible.columns:
            eligible["ev"] = eligible["win_prob"] * eligible["win_odds"]
            out_cols.extend(["win_odds", "ev"])

        result = eligible[out_cols].sort_values("pred_prob", ascending=False).reset_index(drop=True)
        return result

    # method == "ev": 従来の期待値ベースのアプローチ
    out_base = ["horse_num", "horse", "pred_prob", "win_prob", "win_odds", "ev", "bet_amount"]

    if "win_odds" not in df.columns:
        return pd.DataFrame(columns=out_base)

    # 期待値 = win_prob * win_odds
    df["ev"] = df["win_prob"] * df["win_odds"]

    # 最小期待値でフィルタリング
    eligible = df[df["ev"] >= min_ev].copy()

    if eligible.empty:
        return pd.DataFrame(columns=out_base)

    # 単勝確率を用いたケリー方式の賭け金サイジング
    eligible["bet_amount"] = eligible.apply(
        lambda row: compute_bet_amount(
            prob=row["win_prob"],
            odds=row["win_odds"],
            bankroll=bankroll,
            fraction=kelly_frac,
            max_bet_fraction=max_bet_fraction,
            min_bet=min_bet,
        )
        if row["win_odds"] > 0
        else 0.0,
        axis=1,
    )

    out_cols = ["horse_num"]
    if "horse" in eligible.columns:
        out_cols.append("horse")
    out_cols.extend(["pred_prob", "win_prob", "win_odds", "ev", "bet_amount"])

    result = eligible[out_cols].sort_values("ev", ascending=False).reset_index(drop=True)
    return result


def recommend_trio(
    race_feat: pd.DataFrame,
    top_n: int = 5,
    trio_odds_df: pd.DataFrame | None = None,
    min_ev: float = 1.0,
    method: str = "threshold",
) -> pd.DataFrame:
    """Harvilleモデルを用いて三連複を推奨する。

    Args:
        race_feat: 次の列を持つ DataFrame: horse_num, pred_prob, win_odds。
        top_n: 組み合わせ対象とする上位馬の数。
        trio_odds_df: scrape_trio_odds() から取得した DataFrame(任意)。
            列: horse1, horse2, horse3, odds, popularity。
            馬番は1始まり。
        min_ev: 最小期待値閾値(method="ev" かつオッズが利用可能な場合のみ適用)。
        method: "threshold"(閾値方式)または "ev"(従来方式)。

    Returns:
        次の列を持つ DataFrame: horse1, horse2, horse3, trio_prob, odds, ev。
        馬番は1始まり。
    """
    out_cols = ["horse1", "horse2", "horse3", "trio_prob", "odds", "ev"]

    if race_feat.empty or "pred_prob" not in race_feat.columns:
        return pd.DataFrame(columns=out_cols)

    df = race_feat.copy()
    # 取消馬を除外 (netkeiba は取消時に -3.0 を返す)
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0]
    df = df.sort_values("horse_num").reset_index(drop=True)
    horse_nums = df["horse_num"].values  # 1始まりのマッピング

    # 単勝確率を取得
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)

    # 三連複の組み合わせを生成 (0始まり)
    trio_df = generate_trio_combinations(win_probs, top_n=top_n)

    if trio_df.empty:
        return pd.DataFrame(columns=out_cols)

    # 0始まりのインデックスを1始まりの馬番に変換
    trio_df["horse1"] = trio_df["horse1"].map(lambda i: int(horse_nums[i]))
    trio_df["horse2"] = trio_df["horse2"].map(lambda i: int(horse_nums[i]))
    trio_df["horse3"] = trio_df["horse3"].map(lambda i: int(horse_nums[i]))

    # 実際のオッズと突合(可能な場合)
    trio_df["odds"] = np.nan
    trio_df["ev"] = np.nan

    if trio_odds_df is not None and not trio_odds_df.empty:
        # 正規化: 三連複は順序を問わないので馬番をソートしてキーにする
        def _sort_key(row):
            return tuple(sorted([row["horse1"], row["horse2"], row["horse3"]]))

        trio_df["_key"] = trio_df.apply(_sort_key, axis=1)

        odds_lookup = {}
        for _, row in trio_odds_df.iterrows():
            k = tuple(sorted([int(row["horse1"]), int(row["horse2"]), int(row["horse3"])]))
            odds_lookup[k] = row["odds"]

        trio_df["odds"] = trio_df["_key"].map(odds_lookup)
        trio_df["ev"] = trio_df["trio_prob"] * trio_df["odds"]
        trio_df = trio_df.drop(columns=["_key"])

        # オッズがある場合は期待値でフィルタリング (method="ev" のみ)
        if method == "ev":
            has_odds = trio_df["odds"].notna()
            trio_df = trio_df[~has_odds | (trio_df["ev"] >= min_ev)]

    result = trio_df[out_cols].sort_values(
        "trio_prob", ascending=False,
    ).reset_index(drop=True)
    return result


def recommend_trifecta(
    race_feat: pd.DataFrame,
    top_n: int = 5,
    trifecta_odds_df: pd.DataFrame | None = None,
    min_ev: float = 1.0,
    method: str = "threshold",
) -> pd.DataFrame:
    """Harvilleモデルを用いて三連単を推奨する。

    Args:
        race_feat: 次の列を持つ DataFrame: horse_num, pred_prob, win_odds。
        top_n: 組み合わせ対象とする上位馬の数。
        trifecta_odds_df: scrape_trifecta_odds() から取得した DataFrame(任意)。
            列: horse1, horse2, horse3, odds, popularity。
            馬番は1始まり、順序は意味を持つ(1着・2着・3着)。
        min_ev: 最小期待値閾値(method="ev" かつオッズが利用可能な場合のみ適用)。
        method: "threshold"(閾値方式)または "ev"(従来方式)。

    Returns:
        次の列を持つ DataFrame: horse1, horse2, horse3, harville_prob, odds, ev。
        馬番は1始まり、順序は着順。
    """
    out_cols = ["horse1", "horse2", "horse3", "harville_prob", "odds", "ev"]

    if race_feat.empty or "pred_prob" not in race_feat.columns:
        return pd.DataFrame(columns=out_cols)

    df = race_feat.copy()
    # 取消馬を除外 (netkeiba は取消時に -3.0 を返す)
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0]
    df = df.sort_values("horse_num").reset_index(drop=True)
    horse_nums = df["horse_num"].values  # 1始まりのマッピング

    # 単勝確率を取得
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)

    # 三連単の組み合わせを生成 (0始まり)
    trifecta_df = generate_trifecta_combinations(win_probs, top_n=top_n)

    if trifecta_df.empty:
        return pd.DataFrame(columns=out_cols)

    # 0始まりのインデックスを1始まりの馬番に変換
    trifecta_df["horse1"] = trifecta_df["horse1"].map(lambda i: int(horse_nums[i]))
    trifecta_df["horse2"] = trifecta_df["horse2"].map(lambda i: int(horse_nums[i]))
    trifecta_df["horse3"] = trifecta_df["horse3"].map(lambda i: int(horse_nums[i]))

    # 実際のオッズと突合(可能な場合)
    trifecta_df["odds"] = np.nan
    trifecta_df["ev"] = np.nan

    if trifecta_odds_df is not None and not trifecta_odds_df.empty:
        # 三連単は順序が重要なので (h1, h2, h3) をそのままキーにする
        odds_lookup = {}
        for _, row in trifecta_odds_df.iterrows():
            k = (int(row["horse1"]), int(row["horse2"]), int(row["horse3"]))
            odds_lookup[k] = row["odds"]

        def _lookup_odds(row):
            return odds_lookup.get(
                (row["horse1"], row["horse2"], row["horse3"]), np.nan,
            )

        trifecta_df["odds"] = trifecta_df.apply(_lookup_odds, axis=1)
        trifecta_df["ev"] = trifecta_df["harville_prob"] * trifecta_df["odds"]

        # オッズがある場合は期待値でフィルタリング (method="ev" のみ)
        if method == "ev":
            has_odds = trifecta_df["odds"].notna()
            trifecta_df = trifecta_df[~has_odds | (trifecta_df["ev"] >= min_ev)]

    result = trifecta_df[out_cols].sort_values(
        "harville_prob", ascending=False,
    ).reset_index(drop=True)
    return result


def generate_full_recommendation(
    race_feat: pd.DataFrame,
    min_ev: float = 1.0,
    bankroll: float = 1_000_000,
    kelly_frac: float = 0.25,
    top_n: int = 5,
    trio_odds_df: pd.DataFrame | None = None,
    trifecta_odds_df: pd.DataFrame | None = None,
    method: str = "threshold",
    prob_threshold: float = 0.3,
    **tier_kwargs,
) -> dict[str, pd.DataFrame]:
    """全ての馬券種について推奨を生成する。

    Args:
        race_feat: 各馬の予測値とオッズを持つ DataFrame。
            必須列: horse_num, pred_prob。
            任意: horse (馬名), win_odds, show_odds_min, show_odds_max。
        min_ev: 最小期待値閾値(method="ev" 時に使用)。
        bankroll: 現在のバンクロール(method="ev" 時に使用)。
        kelly_frac: ケリー倍率(method="ev" 時に使用)。
        top_n: 三連複・三連単の組み合わせで用いる上位 N 頭。
        trio_odds_df: 三連複オッズの DataFrame(任意)。
        trifecta_odds_df: 三連単オッズの DataFrame(任意)。
        method: "threshold"(閾値方式)または "ev"(従来方式)。
        prob_threshold: 閾値方式での最小 pred_prob。
        **tier_kwargs: compute_tier_bet_amount に渡すティアサイジングパラメータ。

    Returns:
        "show", "win", "trio", "trifecta" をキーとする辞書。各値は推奨 DataFrame。
    """
    result: dict[str, pd.DataFrame] = {}

    # 複勝の推奨
    result["show"] = recommend_show(
        race_feat, min_ev=min_ev, bankroll=bankroll, kelly_frac=kelly_frac,
        method=method, prob_threshold=prob_threshold, **tier_kwargs,
    )

    # 単勝の推奨
    result["win"] = recommend_win(
        race_feat, min_ev=min_ev, bankroll=bankroll, kelly_frac=kelly_frac,
        method=method, prob_threshold=prob_threshold, **tier_kwargs,
    )

    # 三連複の推奨
    result["trio"] = recommend_trio(
        race_feat, top_n=top_n, trio_odds_df=trio_odds_df, min_ev=min_ev,
        method=method,
    )

    # 三連単の推奨
    result["trifecta"] = recommend_trifecta(
        race_feat, top_n=top_n, trifecta_odds_df=trifecta_odds_df, min_ev=min_ev,
        method=method,
    )

    return result
