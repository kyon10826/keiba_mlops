"""Horse selection logic for betting."""

from __future__ import annotations

import pandas as pd


def select_bet_horse(
    race_data: pd.DataFrame,
    prob_col: str = "pred_prob",
    odds_col: str = "win_odds",
    top_n_popularity: int = 3,
    min_expected_value: float = 1.0,
) -> pd.Series | None:
    """Select the best horse to bet on for a single race.

    Strategy:
        1. Compute expected value = calibrated probability * show odds estimate
        2. Filter to horses with EV > min_expected_value
        3. Among top-N popularity, pick the horse with highest jockey_lcb95
        4. Fallback to sire_lcb95 if no jockey match
        5. Skip race if no horse meets EV threshold

    Args:
        race_data: DataFrame for a single race with pred_prob, odds, and stats columns
        prob_col: Column name for calibrated probability
        odds_col: Column name for odds
        top_n_popularity: How many popular horses to consider
        min_expected_value: Minimum expected value threshold

    Returns:
        Series for the selected horse, or None if race should be skipped
    """
    if race_data.empty:
        return None

    rd = race_data.copy()

    # Estimate show odds as ~1/3 of win odds (heuristic)
    rd["_show_odds_est"] = rd[odds_col] / 3.0
    rd["_expected_value"] = rd[prob_col] * rd["_show_odds_est"]

    # Popularity rank within race (lower odds = more popular)
    # Computed on ALL horses before EV filtering to match notebook strategy
    rd["_pop_rank"] = rd[odds_col].rank(method="min")

    # Filter by expected value
    eligible = rd[rd["_expected_value"] >= min_expected_value]
    if eligible.empty:
        return None

    # Top-N popular among eligible (rank computed on full race data)
    top_pop = eligible[eligible["_pop_rank"] <= top_n_popularity]

    if not top_pop.empty and "lcb95_jockey" in top_pop.columns:
        # Pick by highest jockey LCB95
        best_idx = top_pop["lcb95_jockey"].idxmax()
        return race_data.loc[best_idx]

    # Fallback: sire LCB95 among all eligible
    if "lcb95_sire" in eligible.columns:
        best_idx = eligible["lcb95_sire"].idxmax()
        return race_data.loc[best_idx]

    # Last resort: highest predicted probability among eligible
    best_idx = eligible[prob_col].idxmax()
    return race_data.loc[best_idx]


def select_bets_for_all_races(
    df: pd.DataFrame,
    race_key: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Apply selection across all races.

    Returns a DataFrame of selected horses (one per race).
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
    """Select horse using threshold method (Frieren approach).

    Strategy:
        1. Filter horses with pred_prob >= prob_threshold
        2. Among those, check if any are within top-N popularity (by odds rank)
        3. If yes, pick by highest jockey_lcb95
        4. If no jockey match, pick by highest sire_lcb95
        5. If no horse meets threshold, skip race

    Note: Odds are used ONLY for popularity ranking, NOT for EV calculation.

    Args:
        race_data: DataFrame for a single race with pred_prob, odds, and stats columns
        prob_col: Column name for calibrated probability
        odds_col: Column name for odds
        prob_threshold: Minimum predicted probability to consider
        max_popularity: How many popular horses (by odds rank) to consider

    Returns:
        Series for the selected horse, or None if race should be skipped
    """
    # Allow top_n_popularity as alias for max_popularity (config compat)
    if top_n_popularity is not None:
        max_popularity = top_n_popularity

    if race_data.empty:
        return None

    rd = race_data.copy()

    # Filter by probability threshold
    eligible = rd[rd[prob_col] >= prob_threshold]
    if eligible.empty:
        return None

    # Popularity rank within full race (lower odds = more popular)
    rd["_pop_rank"] = rd[odds_col].rank(method="min")
    eligible = eligible.assign(_pop_rank=rd["_pop_rank"])

    # Check for candidates within top-N popularity
    top_pop = eligible[eligible["_pop_rank"] <= max_popularity]

    # Check jockey LCB95 (support both naming conventions)
    jockey_lcb_col = "jockey_lcb95" if "jockey_lcb95" in top_pop.columns else "lcb95_jockey"
    if not top_pop.empty and jockey_lcb_col in top_pop.columns:
        best_idx = top_pop[jockey_lcb_col].idxmax()
        return race_data.loc[best_idx]

    # Fallback: sire LCB95 among all eligible (support both naming conventions)
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
    """Apply threshold selection across all races.

    If total selected horses across all races is less than min_total_candidates,
    the probability threshold is lowered by threshold_step and retried.

    Returns a DataFrame of selected horses (one per race).
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
    """Dispatch to threshold or EV selection based on method.

    Args:
        df: DataFrame with all race data
        method: Selection method - "threshold" for Frieren approach, "ev" for EV-based
        race_key: Columns to group races by
        **kwargs: Additional arguments passed to the selected method

    Returns:
        DataFrame of selected horses
    """
    if method == "threshold":
        return select_bets_for_all_races_threshold(df, race_key=race_key, **kwargs)
    elif method == "ev":
        return select_bets_for_all_races(df, race_key=race_key, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method!r}. Use 'threshold' or 'ev'.")
