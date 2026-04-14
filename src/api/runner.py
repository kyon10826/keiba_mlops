"""Race day execution loop.

Fetches race data via scraping, predicts, and outputs recommendations.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.api.client import RaceDataClient, build_recommendations
from src.scraper.odds import scrape_odds
from src.scraper.race_card import scrape_race_card, scrape_today_races
from src.strategy.kelly import compute_bet_amount
from src.strategy.selector import select_bet_horse


def predict_probs(
    model,
    data: pd.DataFrame,
    feature_columns: list[str],
    calibrator,
) -> np.ndarray:
    """Get calibrated prediction probabilities.

    Handles both sklearn-compatible models (predict_proba) and
    LightGBM Booster (predict).

    Args:
        model: Trained model
        data: Feature DataFrame
        feature_columns: Feature column names
        calibrator: Fitted calibrator

    Returns:
        Array of calibrated probabilities
    """
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(data[feature_columns])[:, 1]
    else:
        raw_probs = model.predict(data[feature_columns])
    return calibrator.predict(raw_probs)


def print_recommendations(recommendations_df: pd.DataFrame) -> None:
    """Print bet recommendations to terminal in a readable table format.

    Args:
        recommendations_df: DataFrame with horse, horse_num, pred_prob,
                            and optionally bet_amount columns
    """
    if recommendations_df.empty:
        print("  No recommendations for this race.")
        return

    has_bet = "bet_amount" in recommendations_df.columns
    print("-" * 70)
    if has_bet:
        print(f"  {'馬番':>4s}  {'馬名':<16s}  {'予測確率':>8s}  {'推奨額':>10s}")
    else:
        print(f"  {'馬番':>4s}  {'馬名':<16s}  {'予測確率':>8s}")
    print("-" * 70)

    for _, row in recommendations_df.iterrows():
        horse_num = str(int(row.get("horse_num", 0))).zfill(2)
        horse = str(row.get("horse", ""))[:16]
        prob = row.get("pred_prob", 0)
        if has_bet:
            bet = row.get("bet_amount", 0)
            print(f"  {horse_num:>4s}  {horse:<16s}  {prob:>8.4f}  {bet:>10,.0f}")
        else:
            print(f"  {horse_num:>4s}  {horse:<16s}  {prob:>8.4f}")

    print("-" * 70)


def run_race_day(
    timetable: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    calibrator,
    feature_columns: list[str],
    client: RaceDataClient,
    cfg: dict,
) -> list[dict]:
    """Execute prediction for an entire race day and output recommendations.

    Fetches real-time odds before each race, predicts win probabilities,
    selects the best bet horse, and prints recommendations.

    Args:
        timetable: Race schedule with place, race_num, start_time
        test_df: Prepared test data with features and race IDs
        model: Trained model (LightGBM Booster or sklearn-compatible)
        calibrator: Fitted calibrator
        feature_columns: Feature column names
        client: Race data client (scraper-based)
        cfg: Config dict

    Returns:
        List of recommendation dicts
    """
    strat = cfg["strategy"]
    bankroll = float(strat["initial_bankroll"])
    pre_race_sec = strat.get("pre_race_seconds", 250)
    results = []

    timetable_sorted = timetable.sort_values("start_time")

    for _, race in timetable_sorted.iterrows():
        place = race["place"]
        race_num = race["race_num"]
        start_time = race["start_time"]

        # Calculate wait time
        target = datetime.strptime(start_time, "%H:%M").replace(
            year=datetime.now().year,
            month=datetime.now().month,
            day=datetime.now().day,
        )
        now = datetime.now()

        if now > target:
            print(f"[{start_time}] {place} R{race_num} - already passed, skipping")
            continue

        wait = (target - now).total_seconds() - pre_race_sec
        if wait < 0:
            print(f"[{start_time}] {place} R{race_num} - too close, skipping")
            continue

        print(f"[{start_time}] {place} R{race_num} - waiting {int(wait)}s...")
        time.sleep(wait)

        # Get race data
        race_data = test_df[
            (test_df["place"] == place) & (test_df["race_num"] == race_num)
        ].copy()

        if race_data.empty:
            print(f"  No data for {place} R{race_num}")
            continue

        # Fetch real-time odds via scraper
        race_id_odds = str(race_data["race_id_odds"].iloc[0])
        odds_df = client.get_odds(race_id_odds)

        if odds_df is not None and not odds_df.empty:
            # Merge scraped odds into race data
            if "horse_num" in odds_df.columns and "odds" in odds_df.columns:
                odds_merge = odds_df[["horse_num", "odds"]].rename(
                    columns={"odds": "odds_rt"}
                )
                race_data = race_data.merge(
                    odds_merge,
                    on="horse_num",
                    how="left",
                )
                race_data["win_odds"] = race_data["odds_rt"].fillna(
                    race_data["win_odds"]
                )

        # Predict (handles both LightGBM Booster and sklearn API)
        race_data["pred_prob"] = predict_probs(
            model, race_data, feature_columns, calibrator
        )

        # Select horse
        best = select_bet_horse(
            race_data,
            top_n_popularity=strat["top_n_popularity"],
            min_expected_value=strat["min_expected_value"],
        )

        if best is None:
            print(f"  No suitable bet for {place} R{race_num}")
            continue

        # Compute recommended bet amount (Kelly)
        show_odds_est = best["win_odds"] / 3.0
        bet_amount = compute_bet_amount(
            best["pred_prob"], show_odds_est, bankroll,
            fraction=strat["kelly_fraction"],
            max_bet_fraction=strat["max_bet_fraction"],
            min_bet=strat["min_bet"],
        )

        if bet_amount <= 0:
            continue

        horse_num_str = str(int(best["horse_num"])).zfill(2)

        rec = {
            "place": place,
            "race_num": race_num,
            "horse": best.get("horse", ""),
            "horse_num": horse_num_str,
            "pred_prob": best["pred_prob"],
            "win_odds": best["win_odds"],
            "bet_amount": bet_amount,
        }
        results.append(rec)

        # Print recommendation
        rec_df = build_recommendations(
            race_data,
            selected_horse_num=int(best["horse_num"]),
            bet_amount=bet_amount,
        )

        print(f"\n=== {place} R{race_num} Recommendations ===")
        print_recommendations(rec_df[rec_df["bet_amount"] > 0])

    if results:
        print(f"\n{'=' * 70}")
        print(f"Total recommendations: {len(results)} races")
        total_bet = sum(r["bet_amount"] for r in results)
        print(f"Total recommended bet: {total_bet:,.0f}")
        print(f"{'=' * 70}")

    return results
