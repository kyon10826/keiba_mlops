"""コールドスタート対策のための確率アンサンブル。

最終的な予測確率を計算するための2つの戦略を提供する:

- **threshold** (閾値方式): キャリブレーション済みのモデル確率を
  そのまま使用する。コールドスタート(未知)の馬については、利用可能な場合、
  ベイズ事前分布(``jockey_lcb95``、``sire_lcb95``)をフォールバックとして使用する。
- **ev** (旧方式): モデルの予測をオッズ由来の確率とブレンドし、
  未知の馬に対しては市場シグナルをより重視する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_implied_probability(odds: pd.Series) -> pd.Series:
    """単勝オッズから正規化されたインプライド確率を計算する。

    生のオッズをインプライド確率に変換し、各レース内で
    確率の合計が1.0になるように正規化する(ブックメーカーの
    オーバーラウンド/控除率を除去する)。

    Args:
        odds: 1レース内の各馬の単勝オッズ。

    Returns:
        各馬の正規化されたインプライド確率。
    """
    raw_implied = 1.0 / odds
    total = raw_implied.sum()
    if total == 0:
        return raw_implied
    return raw_implied / total


def detect_unknown_horses(
    df: pd.DataFrame,
    rolling_cols: list[str] | None = None,
) -> pd.Series:
    """過去のローリング特徴量を持たない馬(コールドスタート)を検出する。

    ローリング特徴量の値がすべてNaNまたは0である場合、その馬は
    「未知」とみなされる。つまり、モデルが参照できるレース履歴が存在しない。

    Args:
        df: ローリング特徴量の列を含むDataFrame。
        rolling_cols: チェック対象の列名。デフォルトは特徴量
            パイプラインの標準的なローリング列。

    Returns:
        Trueが未知の馬を示すBoolean Series。
    """
    if rolling_cols is None:
        rolling_cols = [
            "rank_last",
            "rank_rolling_3",
            "rank_rolling_5",
            "show_rate_last_5",
        ]

    available = [c for c in rolling_cols if c in df.columns]
    if not available:
        return pd.Series(True, index=df.index)

    subset = df[available]
    return (subset.isna() | (subset == 0)).all(axis=1)


def adaptive_blend(
    model_prob: pd.Series,
    implied_prob: pd.Series,
    unknown_mask: pd.Series,
    base_weight: float = 0.7,
) -> pd.Series:
    """モデルの予測とオッズ由来のインプライド確率をブレンドする。

    馬が既知(履歴データあり)か未知(コールドスタート)かによって
    異なる重み付けを使用する:

    - 既知の馬:  base_weight * model + (1 - base_weight) * implied
    - 未知の馬: 0.3 * model + 0.7 * implied  (市場重視)

    Args:
        model_prob: キャリブレーション済みのモデル予測確率。
        implied_prob: オッズ由来のインプライド確率(正規化済み)。
        unknown_mask: Boolean Series (True = 未知/コールドスタートの馬)。
        base_weight: 既知の馬に対するモデルの重み(デフォルト0.7)。

    Returns:
        ブレンドされた最終確率。
    """
    known_weight = base_weight
    unknown_weight = 0.3

    weight = unknown_mask.map({True: unknown_weight, False: known_weight})
    return weight * model_prob + (1 - weight) * implied_prob


def get_final_probability(
    model_prob: pd.Series,
    df: pd.DataFrame,
    method: str = "threshold",
    win_odds: pd.Series | None = None,
    base_weight: float = 0.7,
) -> pd.Series:
    """選択した手法に基づいて最終確率を取得する。

    Args:
        model_prob: キャリブレーション済みのモデル予測確率。
        df: 未知の馬を検出するためのDataFrame。*method*が
            ``"threshold"``の場合、コールドスタートの馬に対して
            ``jockey_lcb95``と``sire_lcb95``列をベイズ事前分布として使用する。
        method: ``"threshold"``はモデル確率をそのまま使用する(閾値方式)、
                ``"ev"``はオッズとブレンドした確率を使用する(旧方式)。
        win_odds: 単勝オッズ(``"ev"``手法では必須)。
        base_weight: ``"ev"``モードで既知の馬に対するモデルの重み。

    Returns:
        最終確率。

    Raises:
        ValueError: *method*が``"ev"``で*win_odds*が指定されていない場合。
    """
    unknown_mask = detect_unknown_horses(df)

    if method == "threshold":
        result = model_prob.copy()

        if not unknown_mask.any():
            return result

        # 未知の馬に対しては、ベイズ事前分布をフォールバックとして使用する
        lcb_cols = ["jockey_lcb95", "sire_lcb95"]
        available_lcb = [c for c in lcb_cols if c in df.columns]

        if available_lcb:
            lcb_values = df.loc[unknown_mask, available_lcb]
            # 利用可能なLCB95事前分布の平均を使用する
            prior = lcb_values.mean(axis=1).astype(np.float64)
            # 事前分布が有効な箇所(NaNでなく、かつ > 0)のみ置き換える
            valid_prior = prior.notna() & (prior > 0)
            result.loc[valid_prior[valid_prior].index] = prior[valid_prior]

        return result

    if method == "ev":
        if win_odds is None:
            raise ValueError("win_odds is required for method='ev'")
        implied_prob = compute_implied_probability(win_odds)
        return adaptive_blend(model_prob, implied_prob, unknown_mask, base_weight)

    raise ValueError(f"Unknown method: {method!r}. Use 'threshold' or 'ev'.")
