"""Real-time odds scraping from netkeiba.

Replaces the competition API (172.192.40.114) with direct
netkeiba scraping for win, show, trio, and trifecta odds.

API type mapping:
    1 = 単勝 (win)      - key: 2-digit horse_num
    2 = 複勝 (show)     - key: 2-digit horse_num
    3 = 馬連 (quinella)  - key: 4-digit pair
    4 = 馬単 (exacta)    - key: 4-digit pair
    5 = ワイド (wide)    - key: 4-digit pair
    6 = 枠連 (bracket)   - key: 4-digit pair
    7 = 三連複 (trio)    - key: 6-digit triplet (sorted)
    8 = 三連単 (trifecta) - key: 6-digit triplet (ordered)
"""

from __future__ import annotations

import logging
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.schema import is_nar_race

logger = logging.getLogger(__name__)

ODDS_BASE = "https://race.netkeiba.com/odds"
NAR_ODDS_BASE = "https://nar.netkeiba.com/odds"


def _request_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
    encoding: str = "EUC-JP",
) -> requests.Response | None:
    """Send GET request with retry logic.

    Args:
        url: Target URL.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.
        encoding: Response encoding.

    Returns:
        Response object or None on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.encoding = encoding
            if resp.status_code == 200:
                return resp
            logger.warning(
                "HTTP %d for %s (attempt %d/%d)",
                resp.status_code, url, attempt + 1, max_retries,
            )
        except requests.RequestException as e:
            logger.warning(
                "Request error for %s (attempt %d/%d): %s",
                url, attempt + 1, max_retries, e,
            )
        if attempt < max_retries - 1:
            time.sleep(interval)
    logger.error("All retries failed for %s", url)
    return None


def _fetch_odds_api(
    race_id: str,
    odds_type: int = 1,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> dict | None:
    """Fetch odds from netkeiba JSON API.

    Args:
        race_id: netkeiba race ID (12 digits).
        odds_type: Odds type (1=win/show, 7=trio, 8=trifecta, etc.).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        Parsed JSON dict or None on failure.
    """
    if is_nar_race(race_id):
        # Try NAR-specific odds API first
        url = (
            f"{NAR_ODDS_BASE.replace('/odds', '')}/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type={odds_type}&action=update"
        )
        resp = _request_with_retry(url, max_retries, timeout, interval, encoding="utf-8")
        if resp is None:
            # Fallback to JRA odds API
            url = (
                f"{ODDS_BASE.replace('/odds', '')}/api/api_get_jra_odds.html"
                f"?race_id={race_id}&type={odds_type}&action=update"
            )
            resp = _request_with_retry(url, max_retries, timeout, interval, encoding="utf-8")
    else:
        url = (
            f"{ODDS_BASE.replace('/odds', '')}/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type={odds_type}&action=update"
        )
        resp = _request_with_retry(url, max_retries, timeout, interval, encoding="utf-8")
    if resp is None:
        return None
    try:
        data = resp.json()
        if data.get("status") in ("result", "middle") and "data" in data:
            return data["data"]
    except Exception as e:
        logger.warning("Failed to parse odds API response for %s: %s", race_id, e)
    return None


def _scrape_win_odds(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> list[dict]:
    """Fetch win (単勝) odds for a race via API.

    Args:
        race_id: netkeiba race ID (12 digits).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        List of dicts with horse_num, win_odds, popularity.
    """
    api_data = _fetch_odds_api(race_id, odds_type=1, max_retries=max_retries,
                               timeout=timeout, interval=interval)
    if not api_data or "odds" not in api_data:
        logger.warning("No odds API data for race %s", race_id)
        return []

    odds_data = api_data["odds"]
    results = []

    # type "1" = win odds: {"horse_num": ["odds", "", "popularity"]}
    win_odds = odds_data.get("1", {})
    for hnum_str, vals in win_odds.items():
        if not isinstance(vals, list) or len(vals) < 3:
            continue
        try:
            horse_num = int(hnum_str)
            odds_val = float(vals[0]) if vals[0] else 0.0
            pop = int(vals[2]) if vals[2] else 0
        except (ValueError, TypeError):
            continue

        results.append({
            "horse_num": horse_num,
            "win_odds": odds_val,
            "popularity": pop,
            "odds_type": "win",
        })

    return results


def _scrape_show_odds(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> list[dict]:
    """Fetch show (複勝) odds for a race via API.

    Args:
        race_id: netkeiba race ID (12 digits).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        List of dicts with horse_num, show_odds_min, show_odds_max.
    """
    api_data = _fetch_odds_api(race_id, odds_type=1, max_retries=max_retries,
                               timeout=timeout, interval=interval)
    if not api_data or "odds" not in api_data:
        return []

    odds_data = api_data["odds"]
    results = []

    # type "2" = show odds: {"horse_num": ["min", "max", "popularity"]}
    show_odds = odds_data.get("2", {})
    for hnum_str, vals in show_odds.items():
        if not isinstance(vals, list) or len(vals) < 2:
            continue
        try:
            horse_num = int(hnum_str)
            show_min = float(vals[0]) if vals[0] else 0.0
            show_max = float(vals[1]) if vals[1] else 0.0
        except (ValueError, TypeError):
            continue

        results.append({
            "horse_num": horse_num,
            "show_odds_min": show_min,
            "show_odds_max": show_max,
            "odds_type": "show",
        })

    return results


def scrape_trio_odds(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 2.0,
) -> pd.DataFrame:
    """Fetch trio (三連複) odds for a race.

    三連複 = 3頭の組み合わせ（順不同）。API type=7。
    キーは6桁の馬番連結（昇順ソート済み、例: "010203"）。

    Args:
        race_id: netkeiba race ID (12 digits).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        DataFrame with columns: horse1, horse2, horse3, odds, popularity.
        Empty DataFrame if scraping fails.
    """
    api_data = _fetch_odds_api(race_id, odds_type=7, max_retries=max_retries,
                               timeout=timeout, interval=interval)
    if not api_data or "odds" not in api_data:
        logger.warning("No trio odds API data for race %s", race_id)
        return pd.DataFrame(columns=["horse1", "horse2", "horse3", "odds", "popularity"])

    odds_data = api_data["odds"].get("7", {})
    results = []

    for key, vals in odds_data.items():
        if not isinstance(vals, list) or len(vals) < 3:
            continue
        if len(key) != 6:
            continue
        try:
            h1 = int(key[0:2])
            h2 = int(key[2:4])
            h3 = int(key[4:6])
            odds_val = float(vals[0]) if vals[0] else 0.0
            pop = int(vals[2]) if vals[2] else 0
        except (ValueError, TypeError):
            continue

        results.append({
            "horse1": h1,
            "horse2": h2,
            "horse3": h3,
            "odds": odds_val,
            "popularity": pop,
        })

    if not results:
        return pd.DataFrame(columns=["horse1", "horse2", "horse3", "odds", "popularity"])

    df = pd.DataFrame(results)
    df = df.sort_values("popularity").reset_index(drop=True)
    logger.info("Scraped trio odds for race %s: %d combinations", race_id, len(df))
    return df


def scrape_trifecta_odds(
    race_id: str,
    horses: list[int] | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 2.0,
) -> pd.DataFrame:
    """Fetch trifecta (三連単) odds for a race.

    三連単 = 3頭の着順指定（順序あり）。API type=8。
    キーは6桁の馬番連結（着順通り、例: "030201" = 1着3番, 2着2番, 3着1番）。

    全通りは膨大（18頭立てで4896通り）なので、horses を指定すると
    その馬番を含む組み合わせのみにフィルタして返す。

    Args:
        race_id: netkeiba race ID (12 digits).
        horses: Optional list of horse numbers to filter.
            If provided, only combinations containing ALL specified horses
            are returned. None returns all combinations.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        DataFrame with columns: horse1, horse2, horse3, odds, popularity.
        horse1=1st, horse2=2nd, horse3=3rd.
        Empty DataFrame if scraping fails.
    """
    api_data = _fetch_odds_api(race_id, odds_type=8, max_retries=max_retries,
                               timeout=timeout, interval=interval)
    if not api_data or "odds" not in api_data:
        logger.warning("No trifecta odds API data for race %s", race_id)
        return pd.DataFrame(columns=["horse1", "horse2", "horse3", "odds", "popularity"])

    odds_data = api_data["odds"].get("8", {})
    results = []

    horse_set = set(horses) if horses else None

    for key, vals in odds_data.items():
        if not isinstance(vals, list) or len(vals) < 3:
            continue
        if len(key) != 6:
            continue
        try:
            h1 = int(key[0:2])
            h2 = int(key[2:4])
            h3 = int(key[4:6])
            odds_val = float(vals[0]) if vals[0] else 0.0
            pop = int(vals[2]) if vals[2] else 0
        except (ValueError, TypeError):
            continue

        # Filter: if horses specified, only include combos containing all of them
        if horse_set and not horse_set.issubset({h1, h2, h3}):
            continue

        results.append({
            "horse1": h1,
            "horse2": h2,
            "horse3": h3,
            "odds": odds_val,
            "popularity": pop,
        })

    if not results:
        return pd.DataFrame(columns=["horse1", "horse2", "horse3", "odds", "popularity"])

    df = pd.DataFrame(results)
    df = df.sort_values("popularity").reset_index(drop=True)
    logger.info(
        "Scraped trifecta odds for race %s: %d combinations%s",
        race_id, len(df),
        f" (filtered by horses {horses})" if horses else "",
    )
    return df


def scrape_odds(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
    include_trio: bool = False,
    include_trifecta: bool = False,
    trifecta_horses: list[int] | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Scrape odds for a race from netkeiba.

    By default, combines win (単勝) and show (複勝) odds into a single
    DataFrame (backward-compatible).

    When include_trio or include_trifecta is True, returns a dict of
    DataFrames keyed by odds type.

    Args:
        race_id: netkeiba race ID (12 digits).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.
        include_trio: If True, also fetch trio (三連複) odds.
        include_trifecta: If True, also fetch trifecta (三連単) odds.
        trifecta_horses: Horse numbers to filter trifecta combos.
            Only used when include_trifecta is True.

    Returns:
        If include_trio and include_trifecta are both False:
            DataFrame with columns: horse_num, win_odds, popularity,
            show_odds_min, show_odds_max.
        Otherwise:
            Dict with keys "win_show" (DataFrame), and optionally
            "trio" (DataFrame), "trifecta" (DataFrame).
    """
    # Scrape win odds
    win_data = _scrape_win_odds(race_id, max_retries, timeout, interval)

    # Delay between requests
    time.sleep(interval)

    # Scrape show odds
    show_data = _scrape_show_odds(race_id, max_retries, timeout, interval)

    if not win_data and not show_data:
        logger.warning("No odds data found for race %s", race_id)
        win_show_df = pd.DataFrame(
            columns=["horse_num", "win_odds", "popularity",
                      "show_odds_min", "show_odds_max"]
        )
    else:
        # Build win odds DataFrame
        if win_data:
            win_df = pd.DataFrame(win_data)[["horse_num", "win_odds", "popularity"]]
        else:
            win_df = pd.DataFrame(columns=["horse_num", "win_odds", "popularity"])

        # Build show odds DataFrame
        if show_data:
            show_df = pd.DataFrame(show_data)[["horse_num", "show_odds_min", "show_odds_max"]]
        else:
            show_df = pd.DataFrame(columns=["horse_num", "show_odds_min", "show_odds_max"])

        # Merge
        if not win_df.empty and not show_df.empty:
            win_show_df = win_df.merge(show_df, on="horse_num", how="outer")
        elif not win_df.empty:
            win_show_df = win_df
            win_show_df["show_odds_min"] = 0.0
            win_show_df["show_odds_max"] = 0.0
        else:
            win_show_df = show_df
            win_show_df["win_odds"] = 0.0
            win_show_df["popularity"] = 0

        # Fill NaN
        win_show_df = win_show_df.fillna(0)

        # Sort by horse number
        win_show_df = win_show_df.sort_values("horse_num").reset_index(drop=True)

    logger.info("Scraped odds for race %s: %d horses", race_id, len(win_show_df))

    # Return simple DataFrame if no extra odds requested (backward-compatible)
    if not include_trio and not include_trifecta:
        return win_show_df

    # Build multi-odds result
    result: dict[str, pd.DataFrame] = {"win_show": win_show_df}

    if include_trio:
        time.sleep(interval)
        result["trio"] = scrape_trio_odds(
            race_id, max_retries=max_retries, timeout=timeout, interval=interval,
        )

    if include_trifecta:
        time.sleep(interval)
        result["trifecta"] = scrape_trifecta_odds(
            race_id, horses=trifecta_horses,
            max_retries=max_retries, timeout=timeout, interval=interval,
        )

    return result
