"""Integrated betting recommendation module.

Combines model predictions, odds data, Harville probabilities, and Kelly
criterion to produce recommendations for show, win, trio, and trifecta bets.
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
    """Return tier label based on predicted probability."""
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
    """Recommend show (複勝) bets based on EV or threshold method.

    Args:
        race_feat: DataFrame with columns: horse_num, horse (name),
            pred_prob, show_odds_min, show_odds_max.
        min_ev: Minimum expected value threshold (used when method="ev").
        bankroll: Current bankroll for Kelly sizing (used when method="ev").
        kelly_frac: Kelly fraction multiplier (used when method="ev").
        max_bet_fraction: Max bet as fraction of bankroll (used when method="ev").
        min_bet: Minimum bet amount (used when method="ev").
        method: "threshold" (Frieren) or "ev" (traditional).
        prob_threshold: Minimum pred_prob for threshold method.
        **tier_kwargs: Tier sizing parameters passed to compute_tier_bet_amount.

    Returns:
        DataFrame sorted by pred_prob (threshold) or EV (ev) descending.
    """
    if race_feat.empty:
        return pd.DataFrame()

    df = race_feat.copy()

    if method == "threshold":
        # Filter by probability threshold
        eligible = df[df["pred_prob"] >= prob_threshold].copy()
        if eligible.empty:
            return pd.DataFrame()

        # Tier-based bet sizing
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

        # Build output columns
        out_cols = ["horse_num"]
        if "horse" in eligible.columns:
            out_cols.append("horse")
        out_cols.extend(["pred_prob", "tier", "bet_amount"])
        # Include show_odds_avg/ev as reference if available
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

    # method == "ev": traditional EV-based approach
    cols = ["horse_num", "pred_prob", "show_odds_avg", "ev", "bet_amount"]

    # Compute average show odds
    if "show_odds_min" in df.columns and "show_odds_max" in df.columns:
        df["show_odds_avg"] = (df["show_odds_min"] + df["show_odds_max"]) / 2.0
    elif "show_odds_avg" in df.columns:
        pass  # already present
    else:
        logger.warning("No show odds columns found, cannot compute show EV")
        return pd.DataFrame(columns=["horse_num", "horse"] + cols[1:])

    # Replace zero/negative/invalid odds (netkeiba returns -3.0 for scratched)
    df["show_odds_avg"] = df["show_odds_avg"].apply(
        lambda x: np.nan if x <= 0 else x
    )

    # EV = pred_prob * show_odds_avg
    df["ev"] = df["pred_prob"] * df["show_odds_avg"]

    # Filter by min EV
    eligible = df[df["ev"] >= min_ev].copy()

    if eligible.empty:
        return pd.DataFrame(columns=["horse_num", "horse"] + cols[1:])

    # Kelly bet sizing
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

    # Select output columns
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
    """Recommend win (単勝) bets based on EV or threshold method.

    Args:
        race_feat: DataFrame with columns: horse_num, horse (name),
            pred_prob, win_odds.
        min_ev: Minimum expected value threshold (used when method="ev").
        bankroll: Current bankroll for Kelly sizing (used when method="ev").
        kelly_frac: Kelly fraction multiplier (used when method="ev").
        max_bet_fraction: Max bet as fraction of bankroll (used when method="ev").
        min_bet: Minimum bet amount (used when method="ev").
        method: "threshold" (Frieren) or "ev" (traditional).
        prob_threshold: Minimum pred_prob for threshold method.
        **tier_kwargs: Tier sizing parameters passed to compute_tier_bet_amount.

    Returns:
        DataFrame sorted by pred_prob (threshold) or EV (ev) descending.
    """
    if race_feat.empty:
        return pd.DataFrame()

    df = race_feat.copy()

    # Exclude scratched/invalid horses if win_odds present
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Estimate win probability from show pred_prob (Harville)
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)
    df["win_prob"] = win_probs

    if method == "threshold":
        # Filter by probability threshold on pred_prob
        eligible = df[df["pred_prob"] >= prob_threshold].copy()
        if eligible.empty:
            return pd.DataFrame()

        # Tier-based bet sizing
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
        # Include win_odds/ev as reference if available
        if "win_odds" in eligible.columns:
            eligible["ev"] = eligible["win_prob"] * eligible["win_odds"]
            out_cols.extend(["win_odds", "ev"])

        result = eligible[out_cols].sort_values("pred_prob", ascending=False).reset_index(drop=True)
        return result

    # method == "ev": traditional EV-based approach
    out_base = ["horse_num", "horse", "pred_prob", "win_prob", "win_odds", "ev", "bet_amount"]

    if "win_odds" not in df.columns:
        return pd.DataFrame(columns=out_base)

    # EV = win_prob * win_odds
    df["ev"] = df["win_prob"] * df["win_odds"]

    # Filter by min EV
    eligible = df[df["ev"] >= min_ev].copy()

    if eligible.empty:
        return pd.DataFrame(columns=out_base)

    # Kelly bet sizing using win probability
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
    """Recommend trio (三連複) bets using Harville model.

    Args:
        race_feat: DataFrame with columns: horse_num, pred_prob, win_odds.
        top_n: Number of top horses to consider for combinations.
        trio_odds_df: Optional DataFrame from scrape_trio_odds()
            with columns: horse1, horse2, horse3, odds, popularity.
            Horse numbers are 1-indexed (馬番).
        min_ev: Minimum EV threshold (only applied when method="ev" and odds available).
        method: "threshold" (Frieren) or "ev" (traditional).

    Returns:
        DataFrame with columns: horse1, horse2, horse3, trio_prob,
        odds, ev. Horse numbers are 1-indexed (馬番).
    """
    out_cols = ["horse1", "horse2", "horse3", "trio_prob", "odds", "ev"]

    if race_feat.empty or "pred_prob" not in race_feat.columns:
        return pd.DataFrame(columns=out_cols)

    df = race_feat.copy()
    # Exclude scratched horses (netkeiba returns -3.0 for scratched)
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0]
    df = df.sort_values("horse_num").reset_index(drop=True)
    horse_nums = df["horse_num"].values  # 1-indexed mapping

    # Get win probabilities
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)

    # Generate trio combinations (0-indexed)
    trio_df = generate_trio_combinations(win_probs, top_n=top_n)

    if trio_df.empty:
        return pd.DataFrame(columns=out_cols)

    # Convert 0-indexed to 1-indexed horse numbers
    trio_df["horse1"] = trio_df["horse1"].map(lambda i: int(horse_nums[i]))
    trio_df["horse2"] = trio_df["horse2"].map(lambda i: int(horse_nums[i]))
    trio_df["horse3"] = trio_df["horse3"].map(lambda i: int(horse_nums[i]))

    # Merge with actual odds if available
    trio_df["odds"] = np.nan
    trio_df["ev"] = np.nan

    if trio_odds_df is not None and not trio_odds_df.empty:
        # Normalize: sort horse numbers for trio (order doesn't matter)
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

        # Filter by EV when odds are available (only for method="ev")
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
    """Recommend trifecta (三連単) bets using Harville model.

    Args:
        race_feat: DataFrame with columns: horse_num, pred_prob, win_odds.
        top_n: Number of top horses to consider for combinations.
        trifecta_odds_df: Optional DataFrame from scrape_trifecta_odds()
            with columns: horse1, horse2, horse3, odds, popularity.
            Horse numbers are 1-indexed, order matters (1st, 2nd, 3rd).
        min_ev: Minimum EV threshold (only applied when method="ev" and odds available).
        method: "threshold" (Frieren) or "ev" (traditional).

    Returns:
        DataFrame with columns: horse1, horse2, horse3, harville_prob,
        odds, ev. Horse numbers are 1-indexed (馬番), order = 着順.
    """
    out_cols = ["horse1", "horse2", "horse3", "harville_prob", "odds", "ev"]

    if race_feat.empty or "pred_prob" not in race_feat.columns:
        return pd.DataFrame(columns=out_cols)

    df = race_feat.copy()
    # Exclude scratched horses (netkeiba returns -3.0 for scratched)
    if "win_odds" in df.columns:
        df = df[df["win_odds"] > 0]
    df = df.sort_values("horse_num").reset_index(drop=True)
    horse_nums = df["horse_num"].values  # 1-indexed mapping

    # Get win probabilities
    pred_prob = df["pred_prob"].values
    win_odds = df["win_odds"].values if "win_odds" in df.columns else np.zeros(len(df))
    win_probs = win_probability_from_show(pred_prob, win_odds)

    # Generate trifecta combinations (0-indexed)
    trifecta_df = generate_trifecta_combinations(win_probs, top_n=top_n)

    if trifecta_df.empty:
        return pd.DataFrame(columns=out_cols)

    # Convert 0-indexed to 1-indexed horse numbers
    trifecta_df["horse1"] = trifecta_df["horse1"].map(lambda i: int(horse_nums[i]))
    trifecta_df["horse2"] = trifecta_df["horse2"].map(lambda i: int(horse_nums[i]))
    trifecta_df["horse3"] = trifecta_df["horse3"].map(lambda i: int(horse_nums[i]))

    # Merge with actual odds if available
    trifecta_df["odds"] = np.nan
    trifecta_df["ev"] = np.nan

    if trifecta_odds_df is not None and not trifecta_odds_df.empty:
        # Trifecta: order matters, so key is (h1, h2, h3) as-is
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

        # Filter by EV when odds are available (only for method="ev")
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
    """Generate recommendations for all bet types.

    Args:
        race_feat: DataFrame with per-horse predictions and odds.
            Required columns: horse_num, pred_prob.
            Optional: horse (name), win_odds, show_odds_min, show_odds_max.
        min_ev: Minimum expected value threshold (used when method="ev").
        bankroll: Current bankroll (used when method="ev").
        kelly_frac: Kelly fraction multiplier (used when method="ev").
        top_n: Top N horses for trio/trifecta combinations.
        trio_odds_df: Optional trio odds DataFrame.
        trifecta_odds_df: Optional trifecta odds DataFrame.
        method: "threshold" (Frieren) or "ev" (traditional).
        prob_threshold: Minimum pred_prob for threshold method.
        **tier_kwargs: Tier sizing parameters passed to compute_tier_bet_amount.

    Returns:
        Dict with keys "show", "win", "trio", "trifecta",
        each containing a recommendation DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}

    # Show recommendations
    result["show"] = recommend_show(
        race_feat, min_ev=min_ev, bankroll=bankroll, kelly_frac=kelly_frac,
        method=method, prob_threshold=prob_threshold, **tier_kwargs,
    )

    # Win recommendations
    result["win"] = recommend_win(
        race_feat, min_ev=min_ev, bankroll=bankroll, kelly_frac=kelly_frac,
        method=method, prob_threshold=prob_threshold, **tier_kwargs,
    )

    # Trio recommendations
    result["trio"] = recommend_trio(
        race_feat, top_n=top_n, trio_odds_df=trio_odds_df, min_ev=min_ev,
        method=method,
    )

    # Trifecta recommendations
    result["trifecta"] = recommend_trifecta(
        race_feat, top_n=top_n, trifecta_odds_df=trifecta_odds_df, min_ev=min_ev,
        method=method,
    )

    return result
