"""Race data client wrapping scraper functions for odds and race card retrieval."""

from __future__ import annotations

import pandas as pd

from src.scraper.odds import scrape_odds
from src.scraper.race_card import scrape_race_card, scrape_today_races


class RaceDataClient:
    """Thin wrapper around scraper functions for race data retrieval."""

    def __init__(self, cfg: dict):
        self.scraper_cfg = cfg.get("scraper", {})

    def get_odds(self, race_id: str) -> pd.DataFrame | None:
        """Fetch odds for a race via scraping.

        Args:
            race_id: Race ID string

        Returns:
            DataFrame with odds data, or None on failure
        """
        try:
            return scrape_odds(race_id)
        except Exception as e:
            print(f"Odds fetch error: {e}")
            return None

    def get_race_card(self, race_id: str) -> pd.DataFrame | None:
        """Fetch race entry table via scraping.

        Args:
            race_id: Race ID string

        Returns:
            DataFrame with race card data, or None on failure
        """
        try:
            return scrape_race_card(race_id)
        except Exception as e:
            print(f"Race card fetch error: {e}")
            return None

    def get_today_races(self, date: str | None = None) -> list[dict]:
        """Fetch today's race schedule via scraping.

        Args:
            date: Date string (optional, defaults to today)

        Returns:
            List of race info dicts
        """
        try:
            return scrape_today_races(date)
        except Exception as e:
            print(f"Today races fetch error: {e}")
            return []


def build_recommendations(
    race_data: pd.DataFrame,
    selected_horse_num: int,
    bet_amount: float,
) -> pd.DataFrame:
    """Build a recommendations DataFrame for a single race.

    Args:
        race_data: DataFrame with horse info and pred_prob
        selected_horse_num: Horse number selected for betting
        bet_amount: Recommended bet amount for the selected horse

    Returns:
        DataFrame with columns: horse, horse_num, pred_prob, bet_amount
    """
    cols = ["horse", "horse_num", "pred_prob"]
    available = [c for c in cols if c in race_data.columns]
    df = race_data[available].copy()
    df["bet_amount"] = 0
    df.loc[df["horse_num"].astype(int) == int(selected_horse_num), "bet_amount"] = (
        bet_amount
    )
    return df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
