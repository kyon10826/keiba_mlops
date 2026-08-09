"""AI 競馬予想マスターズ 2026 の当日データ API / 投票 API クライアント。

当日データ API (大会運営の提供、自己署名証明書のため verify=False):
    GET https://172.192.40.114/data
    headers:
        api-key:      "AI_Keiba_2026" (2026 年大会共通)
        type_of_data: "Racecards" | "Odds"
        id:           Racecards → "YYYYMMDD" / Odds → "YYYYMMDDJJRR" (JJ=場所コード)
    返却:
        Racecards → {"message": "OK", "data": {"timetable": [...], "runtable": [...]}}
        Odds      → {"message": "OK", "data": {"odds_rt": [...]}}
    注意: オッズはサーバが発走 4分30秒前に JRA-VAN から取得するので、
          その少し後 (発走 4分10秒前など) に叩くこと。

投票 API (netkeiba 提供):
    base = https://masters.netkeiba.com/ai2026_student/api
    POST /login  {login_id, password}   → {"status": "OK", "data": {"access_token": ...}}
        トークン有効期限は 5 分。レースごとにログインし直すのが安全。
    POST /bet    Authorization: Bearer <token>
                 {"bet_data": [{"race_id": ..., "mark": {...}, "bet": [{"bet_id": ..., "money": ...}]}]}
    GET  /bet    ?race_id[0]=...        → 投票内容の確認
    POST /logout Authorization: Bearer <token>

race_id の変換 (record_data の 18 桁 = YYYYMMDD PP TT DD RR HH):
    odds 用 12 桁 = YYYYMMDD PP RR       (先頭10桁 + 14:16)
    vote 用 12 桁 = YYYY PP TT DD RR     (先頭4桁 + 8:16)
    ※ 16 桁 (HH なし) を渡しても同じスライスで正しく変換される。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests
import urllib3

logger = logging.getLogger(__name__)

# 自己署名証明書の警告を抑制 (大会運営の指示どおり verify=False で叩く)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_API_URL = "https://172.192.40.114/data"
DATA_API_KEY = "AI_Keiba_2026"
VOTE_API_BASE = "https://masters.netkeiba.com/ai2026_student/api"

PLACE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}


def race_id_to_odds_id(race_id: int | str) -> str:
    """record_data の race_id (16/18桁) → 当日オッズ API 用 12 桁 ID。"""
    s = str(int(race_id))
    return s[0:10] + s[14:16]


def race_id_to_vote_id(race_id: int | str) -> str:
    """record_data の race_id (16/18桁) → 投票 API 用 12 桁 ID (netkeiba 形式)。"""
    s = str(int(race_id))
    return s[0:4] + s[8:16]


class MastersDataClient:
    """当日データ API クライアント (Racecards / 5分前オッズ)。"""

    def __init__(
        self,
        url: str = DATA_API_URL,
        api_key: str = DATA_API_KEY,
        timeout: int = 30,
        max_retries: int = 3,
        retry_interval: float = 5.0,
    ):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def _get(self, type_of_data: str, id_str: str) -> dict:
        headers = {
            "api-key": self.api_key,
            "type_of_data": type_of_data,
            "id": id_str,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    self.url, headers=headers, verify=False, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(
                    "Data API HTTP %d (%s id=%s, attempt %d/%d): %s",
                    resp.status_code, type_of_data, id_str,
                    attempt + 1, self.max_retries, resp.text[:200],
                )
            except requests.RequestException as e:
                last_err = e
                logger.warning(
                    "Data API error (%s id=%s, attempt %d/%d): %s",
                    type_of_data, id_str, attempt + 1, self.max_retries, e,
                )
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_interval)
        raise RuntimeError(
            f"Data API failed after {self.max_retries} attempts "
            f"({type_of_data} id={id_str}): {last_err}"
        )

    def get_racecards(self, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """出走表とタイムテーブルを取得する (朝 9 時までに提供想定)。

        Args:
            date_str: "YYYYMMDD"

        Returns:
            (timetable_df, runtable_df)
            timetable: place, race_num, start_time ("HH:MM") など
            runtable:  record_data と同型の当日出走馬データ (結果列は空)
        """
        data = self._get("Racecards", date_str)
        timetable = pd.DataFrame(data["data"]["timetable"])
        runtable = pd.DataFrame(data["data"]["runtable"])
        # 血統登録番号を record_data と同じ 10 桁に揃える (サンプル準拠)
        if "id" in runtable.columns:
            runtable["id"] = pd.to_numeric(runtable["id"], errors="coerce").fillna(0).astype("int64")
            runtable.loc[runtable["id"] < 1_000_000_000, "id"] += 2_000_000_000
        return timetable, runtable

    def get_odds(self, odds_race_id: str) -> pd.DataFrame:
        """5 分前オッズを取得する (発走 4分30秒前以降に叩く)。

        Args:
            odds_race_id: "YYYYMMDDJJRR" (race_id_to_odds_id で変換した 12 桁)

        Returns:
            odds_rt DataFrame: race_id, odds_type, comb, odds など
            (odds_type=1 が単勝)
        """
        data = self._get("Odds", odds_race_id)
        return pd.DataFrame(data["data"]["odds_rt"])

    def get_win_odds(self, odds_race_id: str) -> pd.DataFrame:
        """単勝オッズのみに絞って返す (comb=馬番ゼロ埋め2桁, odds=float)。"""
        odds = self.get_odds(odds_race_id)
        if odds.empty or "odds_type" not in odds.columns:
            return pd.DataFrame(columns=["comb", "odds"])
        win = odds[odds["odds_type"] == 1].copy()
        win["odds"] = pd.to_numeric(win["odds"], errors="coerce")
        return win[["comb", "odds"]].reset_index(drop=True)


@dataclass
class BetResult:
    """投票 1 回分の結果。

    verified:
        True  = 投票直後の GET /bet で同一 race_id / bet_id / 金額の登録を確認できた
        False = GET /bet に投票内容が見つからない (= プラットフォーム未反映の疑い。
                大会ルール上この投票は無効になりうるので必ず目視確認する)
        None  = 確認自体を行っていない / レスポンス構造が想定外で判定不能
    """
    ok: bool
    race_id_vote: str
    horse_num: int
    amount: int
    remaining_points: int | None
    raw_response: dict
    verified: bool | None = None


def extract_registered_bets(body) -> dict[str, list[dict]]:
    """GET /bet レスポンスから {race_id: [{"bet_id":..., "money":...}]} を抽出する。

    投票確認 API のレスポンス構造は公開仕様が薄いため、JSON を再帰的に走査して
    「race_id を持ち bet リストを含む dict」を拾う寛容なパーサにしている。
    構造が想定外なら空 dict を返す (呼び出し側は verified=None として扱う)。
    """
    found: dict[str, list[dict]] = {}

    def walk(node):
        if isinstance(node, dict):
            rid = node.get("race_id")
            bets = node.get("bet")
            if rid is not None and isinstance(bets, list):
                entry = found.setdefault(str(rid), [])
                for b in bets:
                    if isinstance(b, dict) and "bet_id" in b:
                        entry.append(b)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(body)
    return found


class MastersVoteClient:
    """投票 API クライアント。

    トークン有効期限が 5 分のため、`place_win_bet()` は
    ログイン → 投票 → ログアウト を毎回まとめて行う (公式サンプルと同じ運用)。
    """

    def __init__(
        self,
        login_id: str,
        password: str,
        base_url: str = VOTE_API_BASE,
        timeout: int = 30,
        dry_run: bool = False,
    ):
        self.login_id = login_id
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dry_run = dry_run

    # --- 低レベル API ---

    def login(self) -> str:
        """ログインしてアクセストークンを返す (有効期限 5 分)。"""
        resp = requests.post(
            f"{self.base_url}/login",
            headers={"Content-Type": "application/json"},
            json={"login_id": self.login_id, "password": self.password},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Vote API login failed: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        try:
            return body["data"]["access_token"]
        except (KeyError, TypeError):
            raise RuntimeError(f"Vote API login: unexpected response {body}")

    def logout(self, access_token: str) -> None:
        try:
            requests.post(
                f"{self.base_url}/logout",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.warning("Vote API logout failed (ignored): %s", e)

    def bet(self, bet_data: dict, access_token: str) -> dict:
        """投票を実行する。bet_data は API 仕様の 1 レース分 dict。"""
        resp = requests.post(
            f"{self.base_url}/bet",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            json={"bet_data": [bet_data]},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Vote API bet failed: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        # HTTP 200 でもアプリ層のエラー (残高不足・締切超過など) がありうる。
        # status フィールドが明示的に NG 系ならエラーとして扱う。
        if isinstance(body, dict):
            status = str(body.get("status", "OK")).upper()
            if status not in ("OK", "SUCCESS", "200"):
                raise RuntimeError(f"Vote API bet rejected: {body}")
        return body

    def check_bets(self, race_id_votes: list[str], access_token: str) -> dict:
        """投票内容確認 (GET /bet)。プラットフォーム反映の検証に使う。"""
        params = {f"race_id[{i}]": rid for i, rid in enumerate(race_id_votes)}
        resp = requests.get(
            f"{self.base_url}/bet",
            params=params,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Vote API check failed: HTTP {resp.status_code} {resp.text[:300]}")
        return resp.json()

    # --- 高レベル API ---

    @staticmethod
    def build_win_bet_data(race_id_vote: str, horse_num: int, amount: int) -> dict:
        """単勝 1 点分の bet_data を組み立てる (テスト可能なよう分離)。"""
        horse_str = str(int(horse_num))
        return {
            "race_id": race_id_vote,
            # mark は必須のため形式的に投票馬を指定 (公式サンプル準拠)
            "mark": {horse_str: 1},
            # b1_c0_{馬番} = 単勝
            "bet": [{"bet_id": f"b1_c0_{horse_str}", "money": str(int(amount))}],
        }

    @staticmethod
    def _extract_remaining(body) -> int | None:
        """レスポンスから残ポイントを探す (トップレベル / data 配下の両対応)。"""
        if not isinstance(body, dict):
            return None
        for container in (body, body.get("data") or {}):
            if isinstance(container, dict):
                raw = container.get("remaining_money")
                if raw is not None:
                    try:
                        return int(raw)
                    except (TypeError, ValueError):
                        pass
        return None

    def verify_bet(
        self, race_id_vote: str, horse_num: int, access_token: str,
    ) -> bool | None:
        """GET /bet で当該レースの投票反映を確認する。

        Returns:
            True/False = 反映あり/なし、None = レスポンス構造が想定外で判定不能。
            大会ルール上「API が正常でもプラットフォーム未反映の投票は無効」の
            ため、False の場合は必ず目視確認・再投票の判断を行うこと。
        """
        try:
            body = self.check_bets([race_id_vote], access_token)
        except Exception as e:
            logger.warning("verify_bet: GET /bet failed (%s)", e)
            return None
        registered = extract_registered_bets(body)
        if not registered:
            # 構造が読めなかったのか本当に空なのか区別できない場合は None
            return None if not isinstance(body, dict) else False
        bets = registered.get(str(race_id_vote), [])
        expected_bet_id = f"b1_c0_{int(horse_num)}"
        return any(str(b.get("bet_id")) == expected_bet_id for b in bets)

    def place_win_bet(
        self,
        race_id_vote: str,
        horse_num: int,
        amount: int,
        deadline_ts: float | None = None,
        max_attempts: int = 2,
        retry_interval: float = 3.0,
        verify: bool = True,
    ) -> BetResult:
        """単勝 1 点投票 (ログイン → 投票 → 反映確認 → ログアウトを一括実行)。

        Args:
            race_id_vote: 12 桁の netkeiba 形式 race_id (race_id_to_vote_id で変換)
            horse_num: 馬番 (1-18)
            amount: 投票額 (100 pt 単位)
            deadline_ts: 投票締切の UNIX 時刻。過ぎていたらリトライしない
                (締切 = 発走 3 分前。締切後の投票は無効なので粘らない)
            max_attempts: ログイン〜投票のリトライ回数 (一時的なネットワーク断対策)
            verify: True なら投票直後に GET /bet で反映を確認して verified に載せる
        """
        bet_data = self.build_win_bet_data(race_id_vote, horse_num, amount)

        if self.dry_run:
            logger.info("[DRY-RUN] bet: %s", bet_data)
            return BetResult(
                ok=True, race_id_vote=race_id_vote, horse_num=horse_num,
                amount=amount, remaining_points=None,
                raw_response={"dry_run": True, "bet_data": bet_data},
                verified=None,
            )

        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                access_token = self.login()
                try:
                    body = self.bet(bet_data, access_token)
                    verified = (
                        self.verify_bet(race_id_vote, horse_num, access_token)
                        if verify else None
                    )
                finally:
                    self.logout(access_token)
                if verified is False:
                    logger.warning(
                        "投票は受理されたが GET /bet に未反映 (race=%s)。"
                        "プラットフォームでの目視確認が必要です", race_id_vote,
                    )
                return BetResult(
                    ok=True, race_id_vote=race_id_vote, horse_num=horse_num,
                    amount=amount, remaining_points=self._extract_remaining(body),
                    raw_response=body, verified=verified,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "place_win_bet attempt %d/%d failed (race=%s): %s",
                    attempt + 1, max_attempts, race_id_vote, e,
                )
                if attempt >= max_attempts - 1:
                    break
                if deadline_ts is not None and time.time() + retry_interval >= deadline_ts:
                    logger.error("締切が迫っているためリトライを中止 (race=%s)", race_id_vote)
                    break
                time.sleep(retry_interval)
        raise RuntimeError(
            f"place_win_bet failed (race={race_id_vote}): {last_err}"
        )
