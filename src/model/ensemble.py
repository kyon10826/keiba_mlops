"""Probability ensemble for cold-start mitigation.

Provides two strategies for computing final prediction probabilities:

- **threshold** (Frieren method): Uses calibrated model probabilities
  directly.  For cold-start (unknown) horses, Bayesian priors
  (``jockey_lcb95``, ``sire_lcb95``) are used as fallback when available.
- **ev** (legacy): Blends model predictions with odds-implied
  probabilities, giving more weight to market signals for unknown horses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_implied_probability(odds: pd.Series) -> pd.Series:
    """Compute normalized implied probability from win odds.

    Converts raw odds to implied probabilities and normalizes within
    each race so that probabilities sum to 1.0 (removing the bookmaker
    overround / take rate).

    Args:
        odds: Win odds for horses in a single race.

    Returns:
        Normalized implied probability for each horse.
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
    """Detect horses without historical rolling features (cold-start).

    A horse is considered "unknown" when all its rolling feature values
    are either NaN or 0, meaning the model has no race history to rely on.

    Args:
        df: DataFrame containing rolling feature columns.
        rolling_cols: Column names to check. Defaults to the standard
            rolling columns from the feature pipeline.

    Returns:
        Boolean Series where True indicates an unknown horse.
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
    """Blend model predictions with odds-implied probabilities.

    Uses different weighting depending on whether a horse is known
    (has historical data) or unknown (cold-start):

    - Known horses:  base_weight * model + (1 - base_weight) * implied
    - Unknown horses: 0.3 * model + 0.7 * implied  (market-heavy)

    Args:
        model_prob: Calibrated model prediction probabilities.
        implied_prob: Odds-implied probabilities (normalized).
        unknown_mask: Boolean Series (True = unknown / cold-start horse).
        base_weight: Model weight for known horses (default 0.7).

    Returns:
        Blended final probabilities.
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
    """Get final probability based on selected method.

    Args:
        model_prob: Calibrated model prediction probabilities.
        df: DataFrame for detecting unknown horses.  When *method* is
            ``"threshold"``, columns ``jockey_lcb95`` and ``sire_lcb95``
            are used as Bayesian priors for cold-start horses.
        method: ``"threshold"`` uses model prob directly (Frieren method),
                ``"ev"`` uses odds-blended probability (legacy).
        win_odds: Win odds (required for ``"ev"`` method).
        base_weight: Model weight for known horses in ``"ev"`` mode.

    Returns:
        Final probabilities.

    Raises:
        ValueError: If *method* is ``"ev"`` but *win_odds* is not provided.
    """
    unknown_mask = detect_unknown_horses(df)

    if method == "threshold":
        result = model_prob.copy()

        if not unknown_mask.any():
            return result

        # For unknown horses, use Bayesian priors as fallback
        lcb_cols = ["jockey_lcb95", "sire_lcb95"]
        available_lcb = [c for c in lcb_cols if c in df.columns]

        if available_lcb:
            lcb_values = df.loc[unknown_mask, available_lcb]
            # Use the mean of available LCB95 priors
            prior = lcb_values.mean(axis=1).astype(np.float64)
            # Replace only where prior is valid (not NaN and > 0)
            valid_prior = prior.notna() & (prior > 0)
            result.loc[valid_prior[valid_prior].index] = prior[valid_prior]

        return result

    if method == "ev":
        if win_odds is None:
            raise ValueError("win_odds is required for method='ev'")
        implied_prob = compute_implied_probability(win_odds)
        return adaptive_blend(model_prob, implied_prob, unknown_mask, base_weight)

    raise ValueError(f"Unknown method: {method!r}. Use 'threshold' or 'ev'.")
