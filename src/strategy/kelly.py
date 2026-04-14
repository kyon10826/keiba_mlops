"""Fractional Kelly Criterion and Tier-based sizing for bet amount computation."""

from __future__ import annotations

import numpy as np


def kelly_fraction(prob: float, odds: float) -> float:
    """Compute full Kelly fraction.

    Args:
        prob: Estimated probability of winning
        odds: Decimal odds (payout per unit bet, e.g. 3.0 means 3x return)

    Returns:
        Kelly fraction (can be negative if bet has negative EV)
    """
    b = odds - 1.0  # net odds
    q = 1.0 - prob
    if b <= 0:
        return 0.0
    return (prob * b - q) / b


def compute_bet_amount(
    prob: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
) -> float:
    """Compute bet amount using fractional Kelly criterion.

    Args:
        prob: Estimated probability of show (1-3 finish)
        odds: Estimated show odds (decimal)
        bankroll: Current bankroll
        fraction: Kelly fraction multiplier (0.25 = quarter Kelly)
        max_bet_fraction: Maximum bet as fraction of bankroll
        min_bet: Minimum bet amount (platform minimum)

    Returns:
        Bet amount rounded to nearest 100 (ポイント単位)
    """
    kf = kelly_fraction(prob, odds)

    if kf <= 0:
        return 0.0

    # Apply fractional Kelly
    bet = bankroll * kf * fraction

    # Cap at max fraction of bankroll
    max_bet = bankroll * max_bet_fraction
    bet = min(bet, max_bet)

    # Round to 100-unit increments
    bet = int(bet // 100) * 100

    # Apply minimum
    if bet < min_bet:
        # If Kelly says bet but amount is below minimum, bet minimum
        # (only if EV is positive)
        if kf > 0:
            bet = min_bet
        else:
            bet = 0.0

    return float(bet)


def compute_bet_amounts_batch(
    probs: np.ndarray,
    odds: np.ndarray,
    bankroll: float,
    fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
) -> np.ndarray:
    """Vectorized version of compute_bet_amount."""
    amounts = np.array([
        compute_bet_amount(p, o, bankroll, fraction, max_bet_fraction, min_bet)
        for p, o in zip(probs, odds)
    ])
    return amounts


def compute_tier_bet_amount(
    prob: float,
    tier_low_threshold: float = 0.3,
    tier_mid_threshold: float = 0.4,
    tier_high_threshold: float = 0.5,
    tier_low_amount: float = 100.0,
    tier_mid_amount: float = 300.0,
    tier_high_amount: float = 500.0,
) -> float:
    """Compute bet amount using tier-based sizing (Frieren method).

    Assigns bet amount based on predicted probability tiers:
    - prob >= tier_high_threshold → Buy Aggressive (tier_high_amount)
    - prob >= tier_mid_threshold  → Buy (tier_mid_amount)
    - prob >= tier_low_threshold  → Buy Low (tier_low_amount)
    - prob < tier_low_threshold   → No bet (0)

    Does NOT require odds. Uses only model prediction probability.

    Returns:
        Bet amount (float). 0.0 if below lowest threshold.
    """
    if prob >= tier_high_threshold:
        amount = tier_high_amount
    elif prob >= tier_mid_threshold:
        amount = tier_mid_amount
    elif prob >= tier_low_threshold:
        amount = tier_low_amount
    else:
        return 0.0

    # Round to 100-unit increments
    return float(int(amount // 100) * 100)


def compute_tier_bet_amounts_batch(
    probs: np.ndarray,
    **tier_kwargs,
) -> np.ndarray:
    """Vectorized version of compute_tier_bet_amount."""
    amounts = np.array([
        compute_tier_bet_amount(p, **tier_kwargs)
        for p in probs
    ])
    return amounts


def compute_bet_amount_dispatch(
    prob: float,
    odds: float | None = None,
    bankroll: float | None = None,
    method: str = "tier",
    **kwargs,
) -> float:
    """Dispatch to tier or kelly bet sizing based on method.

    Args:
        prob: Estimated probability of winning.
        odds: Decimal odds (required for kelly method).
        bankroll: Current bankroll (required for kelly method).
        method: "tier" or "kelly".
        **kwargs: Additional keyword arguments passed to the underlying function.

    Returns:
        Bet amount (float).

    Raises:
        ValueError: If method is unknown, or if kelly is selected without odds/bankroll.
    """
    if method == "tier":
        tier_keys = {
            "tier_low_threshold", "tier_mid_threshold", "tier_high_threshold",
            "tier_low_amount", "tier_mid_amount", "tier_high_amount",
        }
        tier_kwargs = {k: v for k, v in kwargs.items() if k in tier_keys}
        return compute_tier_bet_amount(prob, **tier_kwargs)
    elif method == "kelly":
        if odds is None or bankroll is None:
            raise ValueError("kelly method requires both odds and bankroll")
        kelly_keys = {"fraction", "max_bet_fraction", "min_bet"}
        kelly_kwargs = {k: v for k, v in kwargs.items() if k in kelly_keys}
        return compute_bet_amount(prob, odds, bankroll, **kelly_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'tier' or 'kelly'.")
