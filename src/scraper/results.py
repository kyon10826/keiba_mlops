"""Past race results scraping from netkeiba.

Scrapes race result pages from db.netkeiba.com and converts
to the 47-column COLUMN_NAMES format defined in schema.py.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.schema import COLUMN_NAMES, PLACE_CODES, NAR_PLACE_CODES, ALL_PLACE_CODES

logger = logging.getLogger(__name__)

BASE_URL = "https://db.netkeiba.com"

# netkeiba place codes (JRA)
PLACE_CODE_LIST = [
    ("01", "札幌"), ("02", "函館"), ("03", "福島"), ("04", "新潟"),
    ("05", "東京"), ("06", "中山"), ("07", "中京"), ("08", "京都"),
    ("09", "阪神"), ("10", "小倉"),
]

# 地方競馬場コード (NAR)
NAR_PLACE_CODE_LIST = [
    ("30", "門別"), ("31", "北見"), ("32", "岩見沢"), ("33", "帯広"), ("34", "旭川"),
    ("35", "盛岡"), ("36", "水沢"),
    ("42", "浦和"), ("43", "船橋"), ("44", "大井"), ("45", "川崎"),
    ("46", "金沢"), ("47", "笠松"), ("48", "名古屋"),
    ("50", "園田"), ("51", "姫路"),
    ("54", "高知"),
    ("55", "佐賀"),
]

# 全競馬場コードリスト
ALL_PLACE_CODE_LIST = PLACE_CODE_LIST + NAR_PLACE_CODE_LIST


def _request_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> requests.Response | None:
    """Send GET request with retry logic.

    Args:
        url: Target URL.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep interval between retries.

    Returns:
        Response object or None on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.encoding = "EUC-JP"
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


def _parse_time_to_seconds(time_str: str) -> float:
    """Convert race time string 'M:SS.S' to total seconds.

    Examples:
        '1:35.2' -> 95.2
        '59.8' -> 59.8
    """
    time_str = time_str.strip()
    if not time_str or time_str == "--":
        return 0.0
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except (ValueError, IndexError):
        return 0.0


def _parse_weight_inc_dec(weight_str: str) -> tuple[float, float]:
    """Parse horse weight string like '480(+4)' or '480(-2)'.

    Returns:
        (weight, inc_dec) tuple.
    """
    if not weight_str or weight_str.strip() == "":
        return 0.0, 0.0
    weight_str = weight_str.strip()
    m = re.match(r"(\d+)\s*\(([+\-]?\d+)\)", weight_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"(\d+)", weight_str)
    if m2:
        return float(m2.group(1)), 0.0
    return 0.0, 0.0


def _parse_corner_ranks(corner_str: str) -> list[int]:
    """Parse corner passing order string like '3-3-2-2'.

    Returns:
        List of up to 4 corner ranks.
    """
    if not corner_str or corner_str.strip() in ("", "-"):
        return [0, 0, 0, 0]
    parts = re.findall(r"\d+", corner_str)
    ranks = [int(p) for p in parts[:4]]
    while len(ranks) < 4:
        ranks.insert(0, 0)
    return ranks


def _parse_race_header(soup: BeautifulSoup, race_id: str) -> dict[str, Any]:
    """Extract race metadata from the page header.

    Returns:
        Dict with year, month, day, place, dist, track_code, state,
        weather, race_num, times, daily, class_code, corner_num, horse_N, etc.
    """
    info: dict[str, Any] = {}

    # Extract year/place/round/day/race_num from race_id (12 digits)
    # Format: YYYYPPKKDDCC (PP=place, KK=kai, DD=day, CC=race_num)
    rid = str(race_id).strip()
    if len(rid) >= 12:
        info["year"] = int(rid[0:4])
        info["place_code"] = rid[4:6]
        info["times"] = int(rid[6:8])
        info["daily"] = rid[8:10]
        info["race_num"] = int(rid[10:12])
    else:
        info["year"] = 0
        info["place_code"] = "00"
        info["times"] = 0
        info["daily"] = "0"
        info["race_num"] = 0

    # Map place code to place name (JRA + NAR)
    code_to_name = {code: name for code, name in ALL_PLACE_CODE_LIST}
    info["place"] = code_to_name.get(info["place_code"], "")

    # Race data header (distance, track type, condition, weather)
    race_data_elem = soup.find("diary_snap_cut") or soup.find("div", class_="data_intro")
    header_text = ""
    if race_data_elem:
        header_text = race_data_elem.get_text()

    # Also look in the broader race info area
    smalltxt = soup.find("p", class_="smalltxt")
    if smalltxt:
        date_text = smalltxt.get_text()
        # Extract date: "YYYY年MM月DD日"
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_text)
        if date_m:
            info["year"] = int(date_m.group(1))
            info["month"] = int(date_m.group(2))
            info["day"] = int(date_m.group(3))

    if "month" not in info:
        info["month"] = 0
        info["day"] = 0

    # Parse track info from the race title area
    # Look for pattern like "芝1600m" or "ダ1200m"
    race_title_area = soup.find("div", class_="data_intro")
    if race_title_area:
        title_text = race_title_area.get_text()
    else:
        title_text = header_text

    # Distance
    dist_m = re.search(r"(\d{3,4})m", title_text)
    info["dist"] = int(dist_m.group(1)) if dist_m else 0

    # Track type (芝=1, ダート=2)
    if "芝" in title_text:
        track_type = 1
    elif "ダ" in title_text:
        track_type = 2
    else:
        track_type = 0
    info["track_code"] = track_type * 10 + 1

    # Corner count based on distance
    dist = info["dist"]
    if dist <= 1400:
        info["corner_num"] = 2
    elif dist <= 2200:
        info["corner_num"] = 3
    else:
        info["corner_num"] = 4

    # Condition (馬場状態)
    cond_m = re.search(r"[馬場]?\s*[:：]\s*(良|稍重|重|不良)", title_text)
    info["state"] = cond_m.group(1) if cond_m else ""

    # Weather
    weather_m = re.search(r"天候\s*[:：]\s*(\S+)", title_text)
    info["weather"] = weather_m.group(1) if weather_m else ""

    # Class / grade
    info["class_code"] = 0
    grade_keywords = {
        "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
        "オープン": 6, "G3": 7, "G2": 8, "G1": 9,
        "OP": 6, "リステッド": 6, "L": 6,
    }
    for kw, code in grade_keywords.items():
        if kw in title_text:
            info["class_code"] = code
            break

    # Age restriction code
    age_m = re.search(r"(\d)歳以上|(\d)歳", title_text)
    info["age_code"] = 0
    if age_m:
        if age_m.group(1):
            info["age_code"] = int(age_m.group(1))
        elif age_m.group(2):
            info["age_code"] = int(age_m.group(2))

    return info


def _scrape_single_race(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> list[dict[str, Any]]:
    """Scrape a single race result page.

    Args:
        race_id: 12-digit netkeiba race ID (YYYYPPKKDDCC).
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.
        interval: Sleep after successful request.

    Returns:
        List of row dicts in COLUMN_NAMES format.
    """
    url = f"{BASE_URL}/race/{race_id}/"
    resp = _request_with_retry(url, max_retries, timeout, interval)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Parse race header
    race_info = _parse_race_header(soup, race_id)

    # Find the result table
    result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        logger.warning("No result table found for race %s", race_id)
        return []

    rows = result_table.find_all("tr")
    if len(rows) < 2:
        return []

    # Detect column indices from header row (the netkeiba layout has changed
    # over time and now contains extra columns like タイム指数 / スタート指数 etc.)
    header_cells = rows[0].find_all("th")
    header_texts = [h.get_text(strip=True) for h in header_cells]

    def _find_col(*keywords) -> int:
        for i, h in enumerate(header_texts):
            for kw in keywords:
                if kw in h:
                    return i
        return -1

    col_idx = {
        "rank":   _find_col("着順"),
        "waku":   _find_col("枠番", "枠"),
        "hnum":   _find_col("馬番"),
        "horse":  _find_col("馬名"),
        "sex_age": _find_col("性齢"),
        "bweight": _find_col("斤量"),
        "jockey": _find_col("騎手"),
        "time":   _find_col("タイム") if "タイム" in header_texts else 7,
        "diff":   _find_col("着差"),
        "corner": _find_col("通過"),
        "last3f": next(
            (i for i, h in enumerate(header_texts)
             if ("上り" in h or "上がり" in h) and "指数" not in h),
            -1,
        ),
        "odds":   _find_col("単勝"),
        "pop":    _find_col("人気"),
        "weight": _find_col("馬体重"),
        "prize":  _find_col("賞金"),
    }
    # Defaults if header not found (legacy layout)
    defaults = {
        "rank": 0, "waku": 1, "hnum": 2, "horse": 3, "sex_age": 4,
        "bweight": 5, "jockey": 6, "time": 7, "diff": 8,
        "corner": 10, "last3f": 11, "odds": 12, "pop": 13,
        "weight": 14, "prize": 20,
    }
    for k, v in defaults.items():
        if col_idx.get(k, -1) < 0:
            col_idx[k] = v

    # Count horses for field_size
    data_rows = rows[1:]  # skip header
    horse_n = 0
    for row in data_rows:
        cols = row.find_all("td")
        if len(cols) >= 3:
            horse_n += 1

    results = []
    for row in data_rows:
        cols = row.find_all("td")
        if len(cols) < 13:
            continue

        def _safe(idx: int) -> str:
            return cols[idx].get_text(strip=True) if 0 <= idx < len(cols) else ""

        # Rank (着順)
        rank_text = _safe(col_idx["rank"])
        if rank_text.isdigit():
            rank = int(rank_text)
            error_code = 0
        else:
            rank = 0
            error_map = {
                "取消": 1, "除外": 2, "中止": 4, "失格": 5, "降着": 7,
            }
            error_code = 0
            for key, code in error_map.items():
                if key in rank_text:
                    error_code = code
                    break

        # Waku (枠番)
        waku_text = _safe(col_idx["waku"])
        waku_num = int(waku_text) if waku_text.isdigit() else 0

        # Horse number (馬番)
        hnum_text = _safe(col_idx["hnum"])
        horse_num = int(hnum_text) if hnum_text.isdigit() else 0

        # Horse name & ID
        horse_cell = cols[col_idx["horse"]] if 0 <= col_idx["horse"] < len(cols) else None
        horse_link = horse_cell.find("a") if horse_cell else None
        horse_name = horse_link.get_text(strip=True) if horse_link else ""
        horse_id = 0
        if horse_link and "href" in horse_link.attrs:
            m = re.search(r"horse/(\d+)", horse_link["href"])
            if m:
                horse_id = int(m.group(1))

        # Sex/Age
        sex_age_text = _safe(col_idx["sex_age"])
        sex = sex_age_text[0] if sex_age_text else ""
        age_text = sex_age_text[1:] if len(sex_age_text) > 1 else ""
        age = int(age_text) if age_text.isdigit() else 0

        # Basis weight (斤量)
        bw_text = _safe(col_idx["bweight"])
        try:
            basis_weight = float(bw_text)
        except ValueError:
            basis_weight = 0.0

        # Jockey ID
        jockey_cell = cols[col_idx["jockey"]] if 0 <= col_idx["jockey"] < len(cols) else None
        jockey_link = jockey_cell.find("a") if jockey_cell else None
        jockey_id = 0
        if jockey_link and "href" in jockey_link.attrs:
            m = re.search(r"jockey/(?:result/recent/)?(\d+)", jockey_link["href"])
            if m:
                jockey_id = int(m.group(1))

        # Time
        time_text = _safe(col_idx["time"])
        race_time = _parse_time_to_seconds(time_text)

        # Time diff (着差)
        time_diff_text = _safe(col_idx["diff"])

        # Corner passing order (通過)
        corner_text = _safe(col_idx["corner"])
        corner_ranks = _parse_corner_ranks(corner_text)

        # Last 3F
        last3f_text = _safe(col_idx["last3f"])
        try:
            last_3f_time = float(last3f_text)
        except ValueError:
            last_3f_time = 0.0

        # Odds / Popularity
        odds_text = _safe(col_idx["odds"])
        try:
            win_odds = float(odds_text)
        except ValueError:
            win_odds = 0.0

        pop_text = _safe(col_idx["pop"])
        try:
            pop = float(pop_text)
        except ValueError:
            pop = 0.0

        # Weight (馬体重)
        weight_text = _safe(col_idx["weight"])
        weight_val, inc_dec_val = _parse_weight_inc_dec(weight_text)

        # Prize
        prize = 0.0
        prize_text = _safe(col_idx["prize"]).replace(",", "")
        try:
            prize = float(prize_text)
        except ValueError:
            prize = 0.0

        # Build race_id for internal use (derived from netkeiba ID)
        try:
            internal_race_id = int(race_id)
        except ValueError:
            internal_race_id = 0

        record = {
            "race_id": internal_race_id,
            "year": race_info["year"],
            "month": race_info.get("month", 0),
            "day": race_info.get("day", 0),
            "times": race_info["times"],
            "place": race_info["place"],
            "daily": race_info["daily"],
            "race_num": race_info["race_num"],
            "horse": horse_name,
            "jockey_id": jockey_id,
            "horse_N": horse_n,
            "waku_num": waku_num,
            "horse_num": horse_num,
            "class_code": race_info["class_code"],
            "track_code": race_info["track_code"],
            "corner_num": race_info["corner_num"],
            "dist": race_info["dist"],
            "state": race_info.get("state", ""),
            "weather": race_info.get("weather", ""),
            "age_code": race_info.get("age_code", 0),
            "sex": sex,
            "age": age,
            "basis_weight": basis_weight,
            "blinker": "",
            "weight": weight_val,
            "inc_dec": inc_dec_val,
            "weight_code": 0,
            "win_odds": win_odds,
            "rank": rank,
            "time_diff": time_diff_text,
            "time": race_time,
            "corner1_rank": corner_ranks[0],
            "corner2_rank": corner_ranks[1],
            "corner3_rank": corner_ranks[2],
            "corner4_rank": corner_ranks[3],
            "last_3F_time": last_3f_time,
            "last_3F_rank": 0,
            "Ave_3F": 0.0,
            "PCI": 0.0,
            "last_3F_time_diff": 0.0,
            "leg": "",
            "pop": pop,
            "prize": prize,
            "error_code": error_code,
            "father": "",
            "mother": "",
            "id": horse_id,
        }
        results.append(record)

    # Compute last_3F_rank within this race
    valid_3f = [(i, r["last_3F_time"]) for i, r in enumerate(results) if r["last_3F_time"] > 0]
    valid_3f.sort(key=lambda x: x[1])
    for rank_pos, (idx, _) in enumerate(valid_3f, 1):
        results[idx]["last_3F_rank"] = rank_pos

    return results


def _get_cache_path(cache_dir: str, race_id: str) -> str:
    """Return cache file path for a race."""
    return os.path.join(cache_dir, f"{race_id}.csv")


def _is_cached(cache_dir: str, race_id: str) -> bool:
    """Check if race result is already cached."""
    if not cache_dir:
        return False
    return os.path.exists(_get_cache_path(cache_dir, race_id))


def _load_cached(cache_dir: str, race_id: str) -> pd.DataFrame | None:
    """Load cached race result."""
    path = _get_cache_path(cache_dir, race_id)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return None
    return None


def _save_cache(cache_dir: str, race_id: str, df: pd.DataFrame) -> None:
    """Save race result to cache."""
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _get_cache_path(cache_dir, race_id)
    df.to_csv(path, index=False, encoding="utf-8")


def _generate_race_ids(
    year: int,
    month: int | None = None,
    place: str | None = None,
) -> list[str]:
    """Generate candidate race IDs for scraping.

    netkeiba race ID format: YYYYPPKKDDCC
        YYYY: year, PP: place code, KK: kai (1-5), DD: day (1-12), CC: race (01-12)

    Args:
        year: Target year.
        month: If specified, only attempt races around this month.
        place: Place name (e.g. '東京'). If None, try all places.

    Returns:
        List of 12-digit race ID strings to attempt.
    """
    place_codes = []
    if place:
        code = PLACE_CODES.get(place)
        if code:
            place_codes.append(code)
        else:
            logger.warning("Unknown place: %s", place)
            return []
    else:
        place_codes = [code for code, _ in PLACE_CODE_LIST]

    race_ids = []
    for pc in place_codes:
        for kai in range(1, 6):  # 1-5回
            for day in range(1, 13):  # 1-12日
                for race in range(1, 13):  # 1-12R
                    rid = f"{year}{pc}{kai:02d}{day:02d}{race:02d}"
                    race_ids.append(rid)

    return race_ids


def scrape_race_results(
    year: int,
    month: int | None = None,
    place: str | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Scrape past race results from netkeiba.

    Generates candidate race IDs and scrapes each page. Races that
    return no data (invalid IDs) are silently skipped.

    Args:
        year: Target year (e.g. 2024).
        month: Optional month filter. When set, only keep races from that month.
        place: Optional place name filter (e.g. '東京').
        max_retries: Maximum retry attempts per request.
        timeout: Request timeout in seconds.
        interval: Sleep interval between requests (1-2 seconds recommended).
        cache_dir: Directory for caching scraped results. None to disable.

    Returns:
        DataFrame with COLUMN_NAMES columns.
    """
    candidate_ids = _generate_race_ids(year, month, place)
    logger.info(
        "Scraping %d candidate race IDs for year=%d, month=%s, place=%s",
        len(candidate_ids), year, month, place,
    )

    all_records: list[dict] = []
    scraped_count = 0

    for race_id in candidate_ids:
        # Check cache
        if cache_dir and _is_cached(cache_dir, race_id):
            cached = _load_cached(cache_dir, race_id)
            if cached is not None and not cached.empty:
                all_records.extend(cached.to_dict("records"))
                continue

        rows = _scrape_single_race(race_id, max_retries, timeout, interval)

        if rows:
            scraped_count += 1
            all_records.extend(rows)
            logger.info("Scraped race %s: %d horses", race_id, len(rows))

            # Cache result
            if cache_dir:
                race_df = pd.DataFrame(rows)
                _save_cache(cache_dir, race_id, race_df)

        # Respectful delay between requests
        time.sleep(interval)

    logger.info("Scraped %d races, %d total records", scraped_count, len(all_records))

    if not all_records:
        return pd.DataFrame(columns=COLUMN_NAMES)

    df = pd.DataFrame(all_records)

    # Ensure all COLUMN_NAMES exist
    for col in COLUMN_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[COLUMN_NAMES]

    # Filter by month if specified
    if month is not None:
        df = df[df["month"] == month].reset_index(drop=True)

    return df


def save_results(
    df: pd.DataFrame,
    output_path: str,
    encoding: str = "shift_jis",
) -> None:
    """Save scraped results to CSV in the standard format.

    Args:
        df: DataFrame with COLUMN_NAMES columns.
        output_path: Output CSV file path.
        encoding: Output encoding (default: shift_jis for compatibility).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, header=False, encoding=encoding)
    logger.info("Saved %d records to %s", len(df), output_path)
