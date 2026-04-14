"""馬券購入のための馬選定ロジック。"""

from __future__ import annotations

import pandas as pd


def select_bet_horse(
    race_data: pd.DataFrame,
    prob_col: str = "pred_prob",
    odds_col: str = "win_odds",
    top_n_popularity: int = 3,
    min_expected_value: float = 1.0,
) -> pd.Series | None:
    """1レースにおいて最も賭けるべき馬を選定する。

    戦略:
        1. 期待値 = キャリブレーション済み確率 * 複勝オッズ推定値 を計算
        2. 期待値が min_expected_value を超える馬に絞り込む
        3. 人気上位 N 頭の中から jockey_lcb95 が最大の馬を選ぶ
        4. 該当する騎手がいなければ sire_lcb95 にフォールバック
        5. 期待値閾値を満たす馬がなければそのレースをスキップ

    Args:
        race_data: 1レース分の DataFrame。pred_prob, オッズ, 統計列を含む
        prob_col: キャリブレーション済み確率の列名
        odds_col: オッズの列名
        top_n_popularity: 考慮する人気上位馬の数
        min_expected_value: 最小期待値閾値

    Returns:
        選定された馬の Series、レースをスキップする場合は None
    """
    if race_data.empty:
        return None

    rd = race_data.copy()

    # 複勝オッズは単勝オッズの約 1/3 と推定 (ヒューリスティック)
    rd["_show_odds_est"] = rd[odds_col] / 3.0
    rd["_expected_value"] = rd[prob_col] * rd["_show_odds_est"]

    # レース内の人気順位 (オッズが低いほど人気)
    # ノートブック戦略に合わせ、期待値フィルタリング前の全馬に対して計算する
    rd["_pop_rank"] = rd[odds_col].rank(method="min")

    # 期待値でフィルタリング
    eligible = rd[rd["_expected_value"] >= min_expected_value]
    if eligible.empty:
        return None

    # 候補のうち人気上位 N 頭 (人気順位はレース全体で計算済み)
    top_pop = eligible[eligible["_pop_rank"] <= top_n_popularity]

    if not top_pop.empty and "lcb95_jockey" in top_pop.columns:
        # 騎手 LCB95 最大の馬を選ぶ
        best_idx = top_pop["lcb95_jockey"].idxmax()
        return race_data.loc[best_idx]

    # フォールバック: 候補全体の中で種牡馬 LCB95 が最大の馬
    if "lcb95_sire" in eligible.columns:
        best_idx = eligible["lcb95_sire"].idxmax()
        return race_data.loc[best_idx]

    # 最後の手段: 候補の中で予測確率が最大の馬
    best_idx = eligible[prob_col].idxmax()
    return race_data.loc[best_idx]


def select_bets_for_all_races(
    df: pd.DataFrame,
    race_key: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """全レースに対して選定を適用する。

    選定された馬の DataFrame を返す(レースごとに1頭)。
    """
    if race_key is None:
        race_key = ["year", "month", "day", "place", "race_num"]

    results = []
    for _, race_data in df.groupby(race_key):
        horse = select_bet_horse(race_data, **kwargs)
        if horse is not None:
            results.append(horse)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).reset_index(drop=True)


def select_bet_horse_threshold(
    race_data: pd.DataFrame,
    prob_col: str = "pred_prob",
    odds_col: str = "win_odds",
    prob_threshold: float = 0.3,
    max_popularity: int = 3,
    top_n_popularity: int | None = None,
    **_kwargs,
) -> pd.Series | None:
    """閾値方式 (Frieren アプローチ) を用いて馬を選定する。

    戦略:
        1. pred_prob >= prob_threshold を満たす馬に絞り込む
        2. 候補の中で人気上位 N 頭 (オッズ順位) に入る馬があるか確認
        3. あれば jockey_lcb95 最大の馬を選ぶ
        4. 該当する騎手がいなければ sire_lcb95 最大の馬を選ぶ
        5. 閾値を満たす馬がなければレースをスキップ

    注意: オッズは人気順位付けにのみ使用し、期待値計算には使用しない。

    Args:
        race_data: 1レース分の DataFrame。pred_prob, オッズ, 統計列を含む
        prob_col: キャリブレーション済み確率の列名
        odds_col: オッズの列名
        prob_threshold: 対象とする最小予測確率
        max_popularity: 考慮する人気上位馬の数 (オッズ順位基準)

    Returns:
        選定された馬の Series、レースをスキップする場合は None
    """
    # top_n_popularity を max_popularity のエイリアスとして許容 (設定互換性)
    if top_n_popularity is not None:
        max_popularity = top_n_popularity

    if race_data.empty:
        return None

    rd = race_data.copy()

    # 確率閾値でフィルタリング
    eligible = rd[rd[prob_col] >= prob_threshold]
    if eligible.empty:
        return None

    # レース全体の人気順位 (オッズが低いほど人気)
    rd["_pop_rank"] = rd[odds_col].rank(method="min")
    eligible = eligible.assign(_pop_rank=rd["_pop_rank"])

    # 人気上位 N 頭に含まれる候補があるか確認
    top_pop = eligible[eligible["_pop_rank"] <= max_popularity]

    # 騎手 LCB95 を確認 (両方の命名規約をサポート)
    jockey_lcb_col = "jockey_lcb95" if "jockey_lcb95" in top_pop.columns else "lcb95_jockey"
    if not top_pop.empty and jockey_lcb_col in top_pop.columns:
        best_idx = top_pop[jockey_lcb_col].idxmax()
        return race_data.loc[best_idx]

    # フォールバック: 候補全体の中で種牡馬 LCB95 を使用 (両方の命名規約をサポート)
    sire_lcb_col = "sire_lcb95" if "sire_lcb95" in eligible.columns else "lcb95_sire"
    if sire_lcb_col in eligible.columns:
        best_idx = eligible[sire_lcb_col].idxmax()
        return race_data.loc[best_idx]

    return None


def select_bets_for_all_races_threshold(
    df: pd.DataFrame,
    race_key: list[str] | None = None,
    min_total_candidates: int = 30,
    threshold_step: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """全レースに対して閾値方式の選定を適用する。

    選定された馬の合計が min_total_candidates に満たない場合、
    確率閾値を threshold_step ずつ下げて再試行する。

    選定された馬の DataFrame を返す(レースごとに1頭)。
    """
    if race_key is None:
        race_key = ["year", "month", "day", "place", "race_num"]

    prob_threshold = kwargs.pop("prob_threshold", 0.3)
    prob_col = kwargs.get("prob_col", "pred_prob")

    while prob_threshold > 0:
        results = []
        for _, race_data in df.groupby(race_key):
            horse = select_bet_horse_threshold(
                race_data, prob_threshold=prob_threshold, **kwargs
            )
            if horse is not None:
                results.append(horse)

        if len(results) >= min_total_candidates or prob_threshold <= threshold_step:
            break
        prob_threshold -= threshold_step

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).reset_index(drop=True)


def select_bets_dispatch(
    df: pd.DataFrame,
    method: str = "threshold",
    race_key: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """method に応じて閾値方式または期待値方式の選定にディスパッチする。

    Args:
        df: 全レースデータの DataFrame
        method: 選定方式。"threshold" は Frieren アプローチ、"ev" は期待値ベース
        race_key: レースをグループ化する列
        **kwargs: 選定メソッドに渡す追加引数

    Returns:
        選定された馬の DataFrame
    """
    if method == "threshold":
        return select_bets_for_all_races_threshold(df, race_key=race_key, **kwargs)
    elif method == "ev":
        return select_bets_for_all_races(df, race_key=race_key, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method!r}. Use 'threshold' or 'ev'.")
