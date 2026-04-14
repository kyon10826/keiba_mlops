"""netkeibaから出馬表と当日のレース一覧をスクレイピングするモジュール。

もともと src/api/runner.py にあったスクレイピング関数を改良したバージョン。
ここでは runner.py は変更しない。ashigaru2 が後でimportを更新する。
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.schema import PLACE_CODES, NAR_PLACE_CODES, ALL_PLACE_CODES, is_nar_race

_CODE_TO_PLACE = {v: k for k, v in ALL_PLACE_CODES.items()}

logger = logging.getLogger(__name__)

RACE_CARD_BASE = "https://race.netkeiba.com"
NAR_RACE_CARD_BASE = "https://nar.netkeiba.com"
DB_BASE = "https://db.netkeiba.com"


def _request_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
    encoding: str = "EUC-JP",
) -> requests.Response | None:
    """リトライ機能付きでGETリクエストを送信する。

    Args:
        url: 対象のURL。
        max_retries: 最大リトライ回数。
        timeout: リクエストのタイムアウト秒数。
        interval: リトライ間のスリープ間隔。
        encoding: レスポンスのエンコーディング。

    Returns:
        レスポンスオブジェクト。失敗時はNone。
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


def scrape_race_card(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> pd.DataFrame | None:
    """netkeibaから出馬表をスクレイピングする。

    Args:
        race_id: netkeibaのレースID（12桁）。
        max_retries: 最大リトライ回数。
        timeout: リクエストのタイムアウト秒数。
        interval: リトライ間のスリープ間隔。

    Returns:
        出走馬情報を含むDataFrame。失敗時はNone。
    """
    # NAR（地方競馬）レースかどうかを判定
    is_nar = is_nar_race(race_id)

    # まず出馬表ページを試し、過去レースの場合は結果ページにフォールバック
    if is_nar:
        url = f"{NAR_RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
    else:
        url = f"{RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
    resp = _request_with_retry(url, max_retries, timeout, interval)

    is_result_page = False
    if resp is None:
        if is_nar:
            # NAR用フォールバック: race.netkeiba.com を試す
            url = f"{RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
            resp = _request_with_retry(url, max_retries, timeout, interval)
        if resp is None:
            # フォールバック: 結果ページを試す
            url = f"{RACE_CARD_BASE}/race/result.html?race_id={race_id}"
            resp = _request_with_retry(url, max_retries, timeout, interval)
            if resp is None:
                # フォールバック: db.netkeiba.com を試す
                url = f"{DB_BASE}/race/{race_id}/"
                resp = _request_with_retry(url, max_retries, timeout, interval)
                if resp is None:
                    return None
            is_result_page = True

    soup = BeautifulSoup(resp.text, "html.parser")

    # 複数のテーブルクラス名を試す
    # NARページには ShutubaTable と Shutuba_Table の両方があるが、NARの Shutuba_Table
    # は別の予想テーブルなので、まず ShutubaTable を確認する。
    if is_nar:
        horse_table = soup.find("table", class_="ShutubaTable")
        if not horse_table:
            horse_table = soup.find("table", class_="Shutuba_Table")
    else:
        horse_table = soup.find("table", class_="Shutuba_Table")
        if not horse_table:
            horse_table = soup.find("table", class_="ShutubaTable")
    if not horse_table:
        # 結果テーブルのクラス名を試す
        horse_table = soup.find("table", class_="RaceTable01")
        if horse_table:
            is_result_page = True
    if not horse_table:
        horse_table = soup.find("table", id="All_Result_Table")
        if horse_table:
            is_result_page = True
    if not horse_table:
        logger.warning("No race table found for race %s", race_id)
        return None

    race_data = []
    for row in horse_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 10:
            continue

        waku_text = cols[0].get_text(strip=True)
        hnum_text = cols[1].get_text(strip=True)

        # 馬名とID
        horse_link = cols[3].find("a")
        horse_name = horse_link.get_text(strip=True) if horse_link else ""
        horse_id = ""
        if horse_link and "href" in horse_link.attrs:
            m = re.search(r"horse/(\d+)", horse_link["href"])
            if m:
                horse_id = m.group(1)

        # 性別 / 年齢
        sex_age = cols[4].get_text(strip=True)
        sex = sex_age[0] if sex_age else ""
        age = sex_age[1:] if len(sex_age) > 1 else ""

        # 斤量
        weight_text = cols[5].get_text(strip=True)

        # 騎手ID（NARの騎手IDは "a0258" のような英数字の場合がある）
        jockey_link = cols[6].find("a")
        jockey_id = ""
        if jockey_link and "href" in jockey_link.attrs:
            m = re.search(r"jockey/(?:result/recent/)?([a-zA-Z0-9]+)", jockey_link["href"])
            if m:
                raw_id = m.group(1)
                if raw_id.isdigit():
                    jockey_id = raw_id
                else:
                    # 英数字のNAR騎手IDを整数ハッシュに変換
                    jockey_id = str(int(raw_id, 16) if all(c in '0123456789abcdefABCDEF' for c in raw_id) else abs(hash(raw_id)) % (10**9))

        # 調教師（任意、テーブルによっては含まれる）
        trainer = ""
        if len(cols) > 7:
            trainer_link = cols[7].find("a")
            if trainer_link:
                trainer = trainer_link.get_text(strip=True)

        # オッズ（出馬表にあれば取得、通常は cols[9] 以降）
        odds = 0.0
        for c in cols[8:]:
            odds_text = c.get_text(strip=True)
            if re.match(r"\d+\.\d+", odds_text):
                try:
                    odds = float(odds_text)
                    break
                except ValueError:
                    pass

        # 人気
        pop = 0
        for c in cols[8:]:
            pop_text = c.get_text(strip=True)
            if pop_text.isdigit() and 1 <= int(pop_text) <= 30:
                pop = int(pop_text)
                break

        race_data.append({
            "waku_num": int(waku_text) if waku_text.isdigit() else 0,
            "horse_num": int(hnum_text) if hnum_text.isdigit() else 0,
            "horse": horse_name,
            "id": int(horse_id) if horse_id.isdigit() else 0,
            "sex": sex,
            "age": int(age) if age.isdigit() else 0,
            "basis_weight": float(weight_text) if weight_text.replace(".", "").isdigit() else 0.0,
            "jockey_id": int(jockey_id) if jockey_id.isdigit() else 0,
            "trainer": trainer,
            "win_odds": odds,
            "pop": pop,
        })

    if not race_data:
        logger.warning("No horse data parsed for race %s", race_id)
        return None

    df = pd.DataFrame(race_data)

    # --- メタデータの統合 ---

    # 1. race_id をパース（12桁: YYYYPPKKDDRR）
    rid = str(race_id).zfill(12)
    r_year = int(rid[0:4])
    place_code = rid[4:6]
    r_times = int(rid[6:8])
    r_daily = int(rid[8:10])
    r_race_num = int(rid[10:12])

    df["race_id"] = int(race_id)
    df["year"] = r_year
    df["month"] = 0
    df["day"] = 0
    df["times"] = r_times
    df["daily"] = r_daily
    df["race_num"] = r_race_num
    df["place"] = _CODE_TO_PLACE.get(place_code, "")

    # 2. scrape_race_info() からレース情報を取得
    info = scrape_race_info(race_id, max_retries, timeout, interval)
    if info:
        df["dist"] = info["dist"]
        df["track_code"] = info["track_code"]
        df["state"] = info["state"]
        df["weather"] = info["weather"]
        df["race_name"] = info["race_name"]
    else:
        df["dist"] = 0
        df["track_code"] = 0
        df["state"] = ""
        df["weather"] = ""
        df["race_name"] = ""

    # 3. レース名から class_code を判定
    rn = df["race_name"].iloc[0] if len(df) > 0 else ""
    if "G1" in rn or "GI" in rn:
        cc = 100
    elif "G2" in rn or "GII" in rn or "G3" in rn or "GIII" in rn:
        cc = 60
    elif "オープン" in rn or "OP" in rn:
        cc = 40
    elif "条件" in rn or "万下" in rn or "勝クラス" in rn:
        cc = 20
    elif "新馬" in rn or "未勝利" in rn:
        cc = 10
    else:
        cc = 0
    df["class_code"] = cc

    # 4. 計算値
    df["horse_N"] = len(df)
    df["rank"] = 0
    dist_val = df["dist"].iloc[0] if len(df) > 0 else 0
    if dist_val <= 1400:
        cn = 2
    elif dist_val <= 2200:
        cn = 3
    else:
        cn = 4
    df["corner_num"] = cn

    # 5. 父（血統情報）を進捗表示しながら取得
    fathers = []
    total = len(df)
    for i, row in df.iterrows():
        hid = str(row["id"])
        if hid and hid != "0":
            print(f"  血統情報取得中 ({i + 1}/{total}): {row['horse']}...")
            logger.info("Fetching pedigree %d/%d (horse_id=%s)", i + 1, total, hid)
            father, _ = get_horse_pedigree(hid, max_retries, timeout, interval)
            fathers.append(father)
        else:
            fathers.append("")
    df["father"] = fathers

    logger.info("Scraped race card for %s: %d horses (with metadata)", race_id, len(df))
    return df


def scrape_race_info(
    race_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> dict | None:
    """レースのメタデータ（距離、コース、発走時刻、馬場状態、天候）を取得する。

    Args:
        race_id: netkeibaのレースID（12桁）。
        max_retries: 最大リトライ回数。
        timeout: リクエストのタイムアウト秒数。
        interval: リトライ間のスリープ間隔。

    Returns:
        dist, track_code, start_time, state, weather, race_name を含むdict。
        失敗時はNone。
    """
    # NAR（地方競馬）レースかどうかを判定
    is_nar = is_nar_race(race_id)

    if is_nar:
        url = f"{NAR_RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
    else:
        url = f"{RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
    resp = _request_with_retry(url, max_retries, timeout, interval)
    if resp is None and is_nar:
        # フォールバック: JRAのURLを試す
        url = f"{RACE_CARD_BASE}/race/shutuba.html?race_id={race_id}"
        resp = _request_with_retry(url, max_retries, timeout, interval)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    result = {
        "dist": 0,
        "track_code": 0,
        "start_time": "",
        "state": "",
        "weather": "",
        "race_name": "",
    }

    # レース情報エリア
    race_data_elem = soup.find("div", class_="RaceData01")
    if race_data_elem:
        text = race_data_elem.get_text()

        # 距離
        dist_m = re.search(r"(\d{3,4})m", text)
        if dist_m:
            result["dist"] = int(dist_m.group(1))

        # コース種別
        if "芝" in text:
            track_type = 1
        elif "ダ" in text:
            track_type = 2
        else:
            track_type = 0
        result["track_code"] = track_type * 10 + 1

        # 馬場状態
        cond_m = re.search(r"(良|稍重|重|不良)", text)
        if cond_m:
            result["state"] = cond_m.group(1)

        # 天候
        weather_m = re.search(r"天候\s*[:：]\s*(\S+)", text)
        if weather_m:
            result["weather"] = weather_m.group(1)

    # レース情報エリア2（追加情報）
    race_data2 = soup.find("div", class_="RaceData02")
    if race_data2:
        text2 = race_data2.get_text()
        # ここに天候がある場合もある
        if not result["weather"]:
            weather_m = re.search(r"(晴|曇|雨|小雨|雪|小雪)", text2)
            if weather_m:
                result["weather"] = weather_m.group(1)

    # 発走時刻
    time_elem = soup.find("dd", class_="Active")
    if time_elem:
        tm = re.search(r"(\d{1,2}:\d{2})", time_elem.get_text())
        if tm:
            result["start_time"] = tm.group(1)

    # レース名（複数のセレクタを試す）
    race_name_elem = (
        soup.find("div", class_="RaceName")
        or soup.find("h1", class_="RaceName")
        or soup.find("span", class_="RaceName")
    )
    if not race_name_elem:
        main_col = soup.find("div", class_="RaceMainColumn")
        if main_col:
            race_name_elem = main_col.find("h1")
    if race_name_elem:
        result["race_name"] = race_name_elem.get_text(strip=True)

    logger.info(
        "Race info for %s: dist=%d, track=%d, time=%s",
        race_id, result["dist"], result["track_code"], result["start_time"],
    )
    return result


def get_horse_pedigree(
    horse_id: str,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> tuple[str, str]:
    """netkeibaから馬の父と母を取得する。

    Args:
        horse_id: netkeibaの馬ID。
        max_retries: 最大リトライ回数。
        timeout: リクエストのタイムアウト秒数。
        interval: リトライ間のスリープ間隔。

    Returns:
        (父, 母) のタプル。失敗時は空文字列。
    """
    url = f"{DB_BASE}/horse/{horse_id}"
    resp = _request_with_retry(url, max_retries, timeout, interval)
    if resp is None:
        return "", ""

    soup = BeautifulSoup(resp.text, "html.parser")

    father = ""
    mother = ""

    pedigree_table = soup.find("table", class_="blood_table")

    # フォールバック: /horse/ped/ ページを試す（NARの馬に必要）
    if not pedigree_table:
        ped_url = f"{DB_BASE}/horse/ped/{horse_id}/"
        ped_resp = _request_with_retry(ped_url, max_retries, timeout, interval)
        if ped_resp:
            ped_soup = BeautifulSoup(ped_resp.text, "html.parser")
            pedigree_table = ped_soup.find("table", class_="blood_table")
    if pedigree_table:
        # 父 - 最初の b_ml セル
        father_elem = pedigree_table.find("td", class_="b_ml")
        if father_elem:
            link = father_elem.find("a")
            father = link.get_text(strip=True) if link else ""

        # 母 - 4行目の b_ml セル
        rows = pedigree_table.find_all("tr")
        if len(rows) > 3:
            mother_elem = rows[3].find("td", class_="b_ml")
            if mother_elem:
                link = mother_elem.find("a")
                mother = link.get_text(strip=True) if link else ""

    logger.info("Pedigree for horse %s: father=%s, mother=%s", horse_id, father, mother)
    time.sleep(0.5)  # 馬のDBページ用に追加の待機
    return father, mother


def scrape_today_races(
    date: str | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    interval: float = 1.5,
) -> list[dict]:
    """netkeibaから本日のレース一覧をスクレイピングする。

    Args:
        date: 日付文字列 'YYYYMMDD'。省略時は本日。
        max_retries: 最大リトライ回数。
        timeout: リクエストのタイムアウト秒数。
        interval: リトライ間のスリープ間隔。

    Returns:
        race_id, place, race_num, race_name, start_time を含むdictのリスト。
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    url = f"{RACE_CARD_BASE}/top/race_list.html?kaisai_date={date}"
    resp = _request_with_retry(url, max_retries, timeout, interval)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    races = []

    # レース一覧の項目を検索
    race_items = soup.find_all("li", class_="RaceList_DataItem")
    for item in race_items:
        link = item.find("a")
        if not link or "href" not in link.attrs:
            continue

        href = link["href"]
        race_id_m = re.search(r"race_id=(\d+)", href)
        if not race_id_m:
            continue
        race_id = race_id_m.group(1)

        # レース番号
        race_num_elem = item.find("span", class_="Race_Num")
        race_num_text = race_num_elem.get_text(strip=True) if race_num_elem else ""
        race_num_m = re.search(r"(\d+)", race_num_text)
        race_num = int(race_num_m.group(1)) if race_num_m else 0

        # レース名
        race_name_elem = item.find("span", class_="ItemTitle")
        race_name = race_name_elem.get_text(strip=True) if race_name_elem else ""

        # 発走時刻
        time_elem = item.find("span", class_="RaceList_Itemtime")
        start_time = ""
        if time_elem:
            tm = re.search(r"(\d{1,2}:\d{2})", time_elem.get_text())
            if tm:
                start_time = tm.group(1)

        # 開催場所（親の開催ブロックから取得）
        place = ""
        parent_block = item.find_parent("div", class_="RaceList_DataList")
        if parent_block:
            prev = parent_block.find_previous("p", class_="RaceList_DataTitle")
            if prev:
                place_m = re.search(r"(\d+)回(.+?)(\d+)日", prev.get_text())
                if place_m:
                    place = place_m.group(2).strip()

        races.append({
            "race_id": race_id,
            "place": place,
            "race_num": race_num,
            "race_name": race_name,
            "start_time": start_time,
            "date": date,
        })

    logger.info("Found %d races for date %s", len(races), date)
    return races
