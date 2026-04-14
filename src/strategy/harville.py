"""Harville model for trifecta / trio probability estimation."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd


def win_probability_from_show(
    pred_prob: np.ndarray,
    win_odds: np.ndarray,
) -> np.ndarray:
    """Estimate win probabilities from show predictions and win odds.

    Uses implied probability from win odds with overround removal.
    Falls back to normalized pred_prob when odds are unavailable.

    Args:
        pred_prob: Predicted show (top-3) probabilities per horse.
        win_odds: Decimal win odds per horse (e.g. 5.0 means 5x payout).

    Returns:
        Normalized win probability array (sums to 1.0).
    """
    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    win_odds = np.asarray(win_odds, dtype=np.float64)

    # Mask for valid odds (positive, finite)
    valid = np.isfinite(win_odds) & (win_odds > 0)

    if valid.sum() == 0:
        # No valid odds — fallback to normalized pred_prob
        total = pred_prob.sum()
        if total <= 0:
            n = len(pred_prob)
            return np.full(n, 1.0 / n) if n > 0 else pred_prob
        return pred_prob / total

    # Implied probability = 1 / odds (guard against 0 in win_odds)
    safe_odds = np.where(valid, win_odds, 1.0)
    implied = np.where(valid, 1.0 / safe_odds, 0.0)

    win_probs = np.zeros_like(pred_prob, dtype=np.float64)

    if not valid.all():
        # Mixed: some horses have valid odds, some don't
        # Allocate probability mass proportionally
        invalid_pred_sum = pred_prob[~valid].sum()
        valid_pred_sum = pred_prob[valid].sum()
        total_pred = invalid_pred_sum + valid_pred_sum

        if total_pred > 0:
            # Share of total probability for invalid-odds horses
            invalid_share = invalid_pred_sum / total_pred
        else:
            invalid_share = (~valid).sum() / len(valid)

        valid_share = 1.0 - invalid_share

        # Valid horses: normalize implied within their share
        implied_sum = implied[valid].sum()
        if implied_sum > 0:
            win_probs[valid] = implied[valid] / implied_sum * valid_share
        else:
            win_probs[valid] = valid_share / valid.sum()

        # Invalid horses: normalize pred_prob within their share
        if invalid_pred_sum > 0:
            win_probs[~valid] = pred_prob[~valid] / invalid_pred_sum * invalid_share
        else:
            n_invalid = (~valid).sum()
            if n_invalid > 0:
                win_probs[~valid] = invalid_share / n_invalid
    else:
        # All horses have valid odds — simple overround removal
        overround = implied.sum()
        if overround > 0:
            win_probs = implied / overround
        else:
            win_probs = pred_prob / pred_prob.sum() if pred_prob.sum() > 0 else implied

    # Final normalization safety net
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
    """Compute Harville probability P(i=1st, j=2nd, k=3rd).

    Formula:
        P = p_i * (p_j / (1 - p_i)) * (p_k / (1 - p_i - p_j))

    Args:
        win_probs: Win probability array for all horses.
        i: Index of 1st place horse.
        j: Index of 2nd place horse.
        k: Index of 3rd place horse.

    Returns:
        Harville probability for the exact (i, j, k) finish order.
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    p_i = win_probs[i]
    p_j = win_probs[j]
    p_k = win_probs[k]

    # Guard: all probabilities must be positive
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
    """Generate trifecta (三連単) combinations with Harville probabilities.

    Considers top_n horses by win probability and generates all nP3 permutations.

    Args:
        win_probs: Win probability array for all horses.
        top_n: Number of top horses to consider.

    Returns:
        DataFrame with columns (horse1, horse2, horse3, harville_prob),
        sorted by probability descending.
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))

    # Select top-N horses by win probability
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
    """Generate trio (三連複) combinations with Harville probabilities.

    Trio probability = sum of all 6 permutation probabilities for each 3-horse set.

    Args:
        win_probs: Win probability array for all horses.
        top_n: Number of top horses to consider.

    Returns:
        DataFrame with columns (horse1, horse2, horse3, trio_prob),
        sorted by probability descending.
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))

    # Select top-N horses by win probability
    top_indices = np.argsort(win_probs)[::-1][:n]

    rows: list[tuple[int, int, int, float]] = []
    for combo in combinations(sorted(top_indices), 3):
        # Sum over all 6 orderings of the 3 horses
        trio_prob = sum(
            harville_probability(win_probs, i, j, k)
            for i, j, k in permutations(combo)
        )
        rows.append((int(combo[0]), int(combo[1]), int(combo[2]), trio_prob))

    df = pd.DataFrame(rows, columns=["horse1", "horse2", "horse3", "trio_prob"])
    df = df.sort_values("trio_prob", ascending=False).reset_index(drop=True)
    return df
