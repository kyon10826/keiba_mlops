"""三連単・三連複の確率推定のためのHarvilleモデル。"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd


def win_probability_from_show(
    pred_prob: np.ndarray,
    win_odds: np.ndarray,
) -> np.ndarray:
    """複勝予測と単勝オッズから単勝確率を推定する。

    単勝オッズから導かれるインプライド確率を使い、オーバーラウンドを除去する。
    オッズが利用できない場合は正規化した pred_prob にフォールバックする。

    Args:
        pred_prob: 各馬の複勝(3着以内)予測確率。
        win_odds: 各馬の単勝オッズ(小数。例: 5.0 は5倍払戻)。

    Returns:
        正規化された単勝確率の配列(合計1.0)。
    """
    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    win_odds = np.asarray(win_odds, dtype=np.float64)

    # 有効なオッズ(正の有限値)のマスク
    valid = np.isfinite(win_odds) & (win_odds > 0)

    if valid.sum() == 0:
        # 有効なオッズがない場合は正規化した pred_prob にフォールバック
        total = pred_prob.sum()
        if total <= 0:
            n = len(pred_prob)
            return np.full(n, 1.0 / n) if n > 0 else pred_prob
        return pred_prob / total

    # インプライド確率 = 1 / オッズ (win_odds の 0 をガード)
    safe_odds = np.where(valid, win_odds, 1.0)
    implied = np.where(valid, 1.0 / safe_odds, 0.0)

    win_probs = np.zeros_like(pred_prob, dtype=np.float64)

    if not valid.all():
        # 混在: 有効なオッズを持つ馬と持たない馬が混在する場合
        # 確率の総量を比例配分する
        invalid_pred_sum = pred_prob[~valid].sum()
        valid_pred_sum = pred_prob[valid].sum()
        total_pred = invalid_pred_sum + valid_pred_sum

        if total_pred > 0:
            # オッズ無効な馬に割り当てる全体確率のシェア
            invalid_share = invalid_pred_sum / total_pred
        else:
            invalid_share = (~valid).sum() / len(valid)

        valid_share = 1.0 - invalid_share

        # 有効馬: そのシェア内でインプライドを正規化
        implied_sum = implied[valid].sum()
        if implied_sum > 0:
            win_probs[valid] = implied[valid] / implied_sum * valid_share
        else:
            win_probs[valid] = valid_share / valid.sum()

        # 無効馬: そのシェア内で pred_prob を正規化
        if invalid_pred_sum > 0:
            win_probs[~valid] = pred_prob[~valid] / invalid_pred_sum * invalid_share
        else:
            n_invalid = (~valid).sum()
            if n_invalid > 0:
                win_probs[~valid] = invalid_share / n_invalid
    else:
        # 全馬が有効オッズを持つ場合は単純にオーバーラウンドを除去
        overround = implied.sum()
        if overround > 0:
            win_probs = implied / overround
        else:
            win_probs = pred_prob / pred_prob.sum() if pred_prob.sum() > 0 else implied

    # 最終的な正規化(セーフティネット)
    total = win_probs.sum()
    if total > 0:
        win_probs = win_probs / total

    return win_probs


def harville_probability(
    win_probs: np.ndarray,
    i: int,
    j: int,
    k: int,
) -> float:
    """Harville確率 P(i=1着, j=2着, k=3着) を計算する。

    計算式:
        P = p_i * (p_j / (1 - p_i)) * (p_k / (1 - p_i - p_j))

    Args:
        win_probs: 全馬の単勝確率配列。
        i: 1着馬のインデックス。
        j: 2着馬のインデックス。
        k: 3着馬のインデックス。

    Returns:
        正確に (i, j, k) の着順となる Harville 確率。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    p_i = win_probs[i]
    p_j = win_probs[j]
    p_k = win_probs[k]

    # ガード: 全ての確率は正でなければならない
    if p_i <= 0 or p_j <= 0 or p_k <= 0:
        return 0.0

    denom_2nd = 1.0 - p_i
    if denom_2nd <= 0:
        return 0.0

    denom_3rd = 1.0 - p_i - p_j
    if denom_3rd <= 0:
        return 0.0

    return p_i * (p_j / denom_2nd) * (p_k / denom_3rd)


def generate_trifecta_combinations(
    win_probs: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Harville確率付きの三連単の組み合わせを生成する。

    単勝確率の上位 top_n 頭を対象に nP3 の順列を全て列挙する。

    Args:
        win_probs: 全馬の単勝確率配列。
        top_n: 対象とする上位馬の数。

    Returns:
        (horse1, horse2, horse3, harville_prob) の列を持つ DataFrame。
        確率の降順でソートされる。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))

    # 単勝確率の上位 N 頭を選択
    top_indices = np.argsort(win_probs)[::-1][:n]

    rows: list[tuple[int, int, int, float]] = []
    for i, j, k in permutations(top_indices, 3):
        prob = harville_probability(win_probs, i, j, k)
        rows.append((int(i), int(j), int(k), prob))

    df = pd.DataFrame(rows, columns=["horse1", "horse2", "horse3", "harville_prob"])
    df = df.sort_values("harville_prob", ascending=False).reset_index(drop=True)
    return df


def generate_trio_combinations(
    win_probs: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Harville確率付きの三連複の組み合わせを生成する。

    三連複確率 = 3頭の組に対する6通りの順列の確率を全て合計したもの。

    Args:
        win_probs: 全馬の単勝確率配列。
        top_n: 対象とする上位馬の数。

    Returns:
        (horse1, horse2, horse3, trio_prob) の列を持つ DataFrame。
        確率の降順でソートされる。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))

    # 単勝確率の上位 N 頭を選択
    top_indices = np.argsort(win_probs)[::-1][:n]

    rows: list[tuple[int, int, int, float]] = []
    for combo in combinations(sorted(top_indices), 3):
        # 3頭に対する6通りの順序について合計
        trio_prob = sum(
            harville_probability(win_probs, i, j, k)
            for i, j, k in permutations(combo)
        )
        rows.append((int(combo[0]), int(combo[1]), int(combo[2]), trio_prob))

    df = pd.DataFrame(rows, columns=["horse1", "horse2", "horse3", "trio_prob"])
    df = df.sort_values("trio_prob", ascending=False).reset_index(drop=True)
    return df
