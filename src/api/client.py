"""オッズと出馬表を取得するためのスクレイパー関数をラップするレースデータクライアント。"""

from __future__ import annotations

import pandas as pd

from src.scraper.odds import scrape_odds
from src.scraper.race_card import scrape_race_card, scrape_today_races


class RaceDataClient:
    """レースデータ取得のためのスクレイパー関数の薄いラッパー。"""

    def __init__(self, cfg: dict):
        self.scraper_cfg = cfg.get("scraper", {})

    def get_odds(self, race_id: str) -> pd.DataFrame | None:
        """スクレイピングによりレースのオッズを取得する。

        Args:
            race_id: レースID文字列

        Returns:
            オッズデータを含むDataFrame、失敗時はNone
        """
        try:
            return scrape_odds(race_id)
        except Exception as e:
            print(f"Odds fetch error: {e}")
            return None

    def get_race_card(self, race_id: str) -> pd.DataFrame | None:
        """スクレイピングにより出馬表を取得する。

        Args:
            race_id: レースID文字列

        Returns:
            出馬表データを含むDataFrame、失敗時はNone
        """
        try:
            return scrape_race_card(race_id)
        except Exception as e:
            print(f"Race card fetch error: {e}")
            return None

    def get_today_races(self, date: str | None = None) -> list[dict]:
        """スクレイピングにより本日のレーススケジュールを取得する。

        Args:
            date: 日付文字列（省略可、デフォルトは本日）

        Returns:
            レース情報の辞書リスト
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
    """単一レースの推奨情報DataFrameを構築する。

    Args:
        race_data: 馬情報とpred_probを含むDataFrame
        selected_horse_num: 購入対象に選択された馬番
        bet_amount: 選択された馬に対する推奨購入金額

    Returns:
        カラム horse, horse_num, pred_prob, bet_amount を持つDataFrame
    """
    cols = ["horse", "horse_num", "pred_prob"]
    available = [c for c in cols if c in race_data.columns]
    df = race_data[available].copy()
    df["bet_amount"] = 0
    df.loc[df["horse_num"].astype(int) == int(selected_horse_num), "bet_amount"] = (
        bet_amount
    )
    return df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
