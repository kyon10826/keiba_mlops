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

    @staticmethod
    def _unwrap_data(body: dict, key: str) -> list:
        """レスポンスの data 部分から `key` (timetable/runtable/odds_rt) を取り出す。

        API が返す形式は次のいずれかを想定 (運営側の実装ゆらぎに対応):
          A. {"data": {"timetable": [...], "runtable": [...]}}  ← 従来
          B. {"data": [{"timetable": [...], "runtable": [...]}]} ← 9/5 の実観測
          C. {"data": [{"key": [...]}]}                          ← key ごとにラップ
        いずれの場合も key に対応するリストを返す。見つからなければ [] を返す。
        """
        d = body.get("data") if isinstance(body, dict) else None
        if d is None:
            return []
        # A: 直接 dict
        if isinstance(d, dict) and key in d:
            v = d[key]
            return v if isinstance(v, list) else []
        # B/C: list ラップ
        if isinstance(d, list):
            for elem in d:
                if isinstance(elem, dict) and key in elem:
                    v = elem[key]
                    if isinstance(v, list):
                        return v
        return []

    def get_racecards(self, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """出走表とタイムテーブルを取得する (朝 9 時までに提供想定)。

        Args:
            date_str: "YYYYMMDD"

        Returns:
            (timetable_df, runtable_df)
            timetable: place, race_num, start_time ("HH:MM") など
            runtable:  record_data と同型の当日出走馬データ (結果列は空)

        レスポンス形式は _unwrap_data 参照 (dict / list ラップの両対応)。
        """
        data = self._get("Racecards", date_str)
        timetable = pd.DataFrame(self._unwrap_data(data, "timetable"))
        runtable = pd.DataFrame(self._unwrap_data(data, "runtable"))
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
        return pd.DataFrame(self._unwrap_data(data, "odds_rt"))

    def get_win_odds(self, odds_race_id: str) -> pd.DataFrame:
        """単勝オッズのみに絞って返す (comb=馬番ゼロ埋め2桁, odds=float)。"""
        odds = self.get_odds(odds_race_id)
        if odds.empty or "odds_type" not in odds.columns:
            return pd.DataFrame(columns=["comb", "odds"])
        win = odds[odds["odds_type"] == 1].copy()
        win["odds"] = pd.to_numeric(win["odds"], errors="coerce")
        return win[["comb", "odds"]].reset_index(drop=True)

    def get_all_odds(self, odds_race_id: str) -> pd.DataFrame:
        """全券種のオッズを返す (odds_type: 1単勝, 2複勝, 3枠連, 4馬連, 5ワイド, 6馬単, 7三連複, 8三連単)。

        multi_bet.select_multi_bets が期待する形式:
            列 comb (str), odds_type (int), odds (float)
        取得失敗時は空 DataFrame。
        """
        try:
            odds = self.get_odds(odds_race_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("get_all_odds 失敗 race=%s (%s)", odds_race_id, e)
            return pd.DataFrame(columns=["comb", "odds_type", "odds"])
        if odds.empty or "odds_type" not in odds.columns:
            return pd.DataFrame(columns=["comb", "odds_type", "odds"])
        out = odds.copy()
        out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
        out = out.dropna(subset=["odds"])
        out = out[out["odds"] > 0]
        return out[["comb", "odds_type", "odds"]].reset_index(drop=True)


# 当日出馬表 API が返す 14 列 (データ仕様書「出馬表(当日API)」準拠)
RUNTABLE_API_COLUMNS = [
    "place", "race_num", "horse_num", "dist", "horse", "sex", "age",
    "jockey", "loaf_weight", "father", "mother", "id", "waku_num", "race_id",
]

# 朝の時点で不明な列の補完値 (record_data の最頻値)
DEFAULT_STATE = "良"
DEFAULT_WEATHER = "晴"


def normalize_runtable(runtable: pd.DataFrame, timetable: pd.DataFrame) -> pd.DataFrame:
    """当日出馬表 API (14 列) を record_data スキーマ (47 列) に正規化する。

    実 API のレスポンスは学習データ (record_data) と列構成が大きく異なる:
      - 騎手は「騎手名 (TARGET 仕様 4 文字)」のみで jockey_id が無い
        → ここでは 0 のまま。run_live 側で netkeiba から ID を補う
      - class_code / track_code / year / month / day は timetable 側にある
        → (place, race_num) で結合
      - 斤量は loaf_weight という列名 → basis_weight に改名
      - 馬体重 (weight/inc_dec)・馬場状態・天候は朝の時点では未公開
        → weight は NaN (run_live 側で前走馬体重を補完)、
          state/weather は最頻値で補完 (発走前に判明すれば上書き)
      - 結果系の列 (rank, time, pop, prize, ...) は 0 で埋める。
        ローリング特徴量は全て shift(1) なので当該行の 0 は特徴量に混入しない

    リプレイ (record_data 由来で 47 列揃っている) を渡した場合は無変更で返す。
    """
    from src.data.schema import COLUMN_NAMES

    out = runtable.copy()
    if all(c in out.columns for c in ("jockey_id", "class_code", "track_code", "year")):
        return out  # 既に record_data スキーマ

    # timetable から開催メタ情報を結合
    tt_cols = [c for c in ("year", "month", "day", "class_code", "track_code")
               if c in timetable.columns]
    tt = timetable[["place", "race_num"] + tt_cols].drop_duplicates(["place", "race_num"])
    out = out.merge(tt, on=["place", "race_num"], how="left", suffixes=("", "_tt"))

    rid = out["race_id"].astype("int64").astype(str)
    # 年は record_data 慣習の 2 桁 (loader が 4 桁化する経路と同じ扱いになる)
    if "year" not in out.columns or out["year"].isna().any():
        out["year"] = rid.str[0:4].astype(int)
    out["year"] = out["year"].astype(int) % 100
    if "month" not in out.columns or out["month"].isna().any():
        out["month"] = rid.str[4:6].astype(int)
    if "day" not in out.columns or out["day"].isna().any():
        out["day"] = rid.str[6:8].astype(int)
    out["times"] = rid.str[10:12].astype(int)
    out["daily"] = rid.str[12:14].astype(int)

    if "loaf_weight" in out.columns:
        out["basis_weight"] = pd.to_numeric(out["loaf_weight"], errors="coerce")
    out["horse_N"] = out.groupby(["place", "race_num"])["horse_num"].transform("count")

    if "jockey_id" not in out.columns:
        out["jockey_id"] = 0
    if "state" not in out.columns:
        out["state"] = DEFAULT_STATE
    if "weather" not in out.columns:
        out["weather"] = DEFAULT_WEATHER
    if "blinker" not in out.columns:
        out["blinker"] = ""
    if "weight" not in out.columns:
        out["weight"] = float("nan")
    if "inc_dec" not in out.columns:
        out["inc_dec"] = float("nan")

    for col in COLUMN_NAMES:
        if col not in out.columns:
            out[col] = 0
    for col in ("class_code", "track_code"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


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
        # 仕様 (2026 参加マニュアル): status=NG なら 1 件も処理されない。
        # status=OK でも data.error_count > 0 / success_count == 0 は失敗扱い。
        if isinstance(body, dict):
            status = str(body.get("status", "OK")).upper()
            if status not in ("OK", "SUCCESS", "200"):
                raise RuntimeError(f"Vote API bet rejected: {body}")
            data = body.get("data")
            if isinstance(data, dict):
                err_n = data.get("error_count")
                ok_n = data.get("success_count")
                if (err_n is not None and int(err_n) > 0) or (ok_n is not None and int(ok_n) == 0):
                    raise RuntimeError(
                        f"Vote API bet not accepted (success={ok_n}, error={err_n}, "
                        f"list_error={data.get('list_error')})")
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

    def verify_many(self, items: list[tuple[str, int, int]]) -> dict[str, bool | None]:
        """複数レースの投票反映をまとめて確認する (ログイン 1 回)。

        Args:
            items: [(race_id_vote, horse_num, amount), ...]
        Returns:
            {race_id_vote: True (一致) / False (未反映 or 不一致) / None (判定不能)}

        仕様上、投票直後は非同期処理のため未反映になりうる (1 分程度あける推奨)。
        呼び出し側は投票から 60 秒以上経ったものだけを渡すこと。
        """
        if not items:
            return {}
        result: dict[str, bool | None] = {}
        try:
            token = self.login()
        except Exception as e:  # noqa: BLE001
            logger.warning("verify_many: login failed (%s)", e)
            return {rid: None for rid, _, _ in items}
        try:
            body = self.check_bets([rid for rid, _, _ in items], token)
        except Exception as e:  # noqa: BLE001
            logger.warning("verify_many: GET /bet failed (%s)", e)
            return {rid: None for rid, _, _ in items}
        finally:
            self.logout(token)
        registered = extract_registered_bets(body)
        for rid, horse_num, amount in items:
            bets = registered.get(str(rid), [])
            expected = f"b1_c0_{int(horse_num)}"
            ok = False
            for b in bets:
                if str(b.get("bet_id")) == expected:
                    money = pd.to_numeric(b.get("money"), errors="coerce")
                    ok = bool(pd.notna(money) and int(money) == int(amount))
                    break
            result[str(rid)] = ok
        return result

    @staticmethod
    def build_multi_bet_data(
        race_id_vote: str, bets: list, use_padded_bet_id: bool = False,
    ) -> dict:
        """複数買い目を 1 レース分の bet_data にまとめる。

        Args:
            race_id_vote: 12 桁 race_id
            bets: MultiBetCandidate のリスト (bet_id / amount を持つ)
            use_padded_bet_id: True なら ゼロ埋め形式 (b7_c0_010203)、
                False なら ハイフン形式 (b7_c0_1-2-3)。初回投票で NG が返る場合
                config で切替できるように両対応。

        Returns:
            {"race_id": ..., "mark": {...}, "bet": [{bet_id, money}, ...]}
        """
        first_horse = str(int(bets[0].horses[0])) if bets else "1"
        return {
            "race_id": race_id_vote,
            "mark": {first_horse: 1},  # mark は形式的な必須項目
            "bet": [
                {
                    "bet_id": (b.bet_id_padded if use_padded_bet_id else b.bet_id),
                    "money": str(int(b.amount)),
                }
                for b in bets
            ],
        }

    def place_multi_bet(
        self,
        race_id_vote: str,
        bets: list,
        deadline_ts: float | None = None,
        max_attempts: int = 2,
        retry_interval: float = 3.0,
        use_padded_bet_id: bool = False,
        split: bool = True,
    ):
        """複数買い目を送信する。

        Args:
            split: True (既定) = 1 買い目ずつ独立に POST (login 1 回で使い回し)。
                   ある買い目の bet_id 形式が NG でも他は成立する故障分離型。
                   False = 全買い目を 1 つの bet_data 配列にまとめて 1 POST。
                   仕様上「エラー時は 1 件も処理されない」ので、format 不明な
                   多券種を混ぜる場合は split=True 推奨。

        Returns:
            split=False: 単一 BetResult (集約)
            split=True:  list[BetResult | None]  買い目ごとの結果 (None = 失敗)
        """
        if not bets:
            return None

        # --- split=False: 従来のバッチ送信 ---
        if not split:
            bet_data = self.build_multi_bet_data(race_id_vote, bets, use_padded_bet_id)
            total_amount = sum(int(b.amount) for b in bets)
            if self.dry_run:
                logger.info("[DRY-RUN] multi bet (batch): %s", bet_data)
                return BetResult(
                    ok=True, race_id_vote=race_id_vote, horse_num=0,
                    amount=total_amount, remaining_points=None,
                    raw_response={"dry_run": True, "bet_data": bet_data}, verified=None,
                )
            last_err: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    access_token = self.login()
                    try:
                        body = self.bet(bet_data, access_token)
                    finally:
                        self.logout(access_token)
                    return BetResult(
                        ok=True, race_id_vote=race_id_vote, horse_num=0,
                        amount=total_amount,
                        remaining_points=self._extract_remaining(body),
                        raw_response=body, verified=None,
                    )
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    logger.warning(
                        "place_multi_bet(batch) attempt %d/%d failed (race=%s): %s",
                        attempt + 1, max_attempts, race_id_vote, e,
                    )
                    if attempt >= max_attempts - 1:
                        break
                    if deadline_ts is not None and time.time() + retry_interval >= deadline_ts:
                        logger.error("締切迫るためリトライ中止 (race=%s)", race_id_vote)
                        break
                    time.sleep(retry_interval)
            raise RuntimeError(f"place_multi_bet(batch) failed (race={race_id_vote}): {last_err}")

        # --- split=True: 買い目ごとに独立 POST (login 1 回で使い回し) ---
        results: list[BetResult | None] = []
        if self.dry_run:
            for cand in bets:
                bet_id = cand.bet_id_padded if use_padded_bet_id else cand.bet_id
                bd = {
                    "race_id": race_id_vote,
                    "mark": {str(int(cand.horses[0])): 1},
                    "bet": [{"bet_id": bet_id, "money": str(int(cand.amount))}],
                }
                logger.info("[DRY-RUN] multi bet (split): %s", bd)
                results.append(BetResult(
                    ok=True, race_id_vote=race_id_vote, horse_num=int(cand.horses[0]),
                    amount=int(cand.amount), remaining_points=None,
                    raw_response={"dry_run": True, "bet_data": bd}, verified=None,
                ))
            return results

        access_token = self.login()
        try:
            for cand in bets:
                if deadline_ts is not None and time.time() >= deadline_ts:
                    logger.error("締切超過のため以降の buy 中止 (race=%s)", race_id_vote)
                    results.append(None)
                    continue
                bet_id = cand.bet_id_padded if use_padded_bet_id else cand.bet_id
                bd = {
                    "race_id": race_id_vote,
                    "mark": {str(int(cand.horses[0])): 1},
                    "bet": [{"bet_id": bet_id, "money": str(int(cand.amount))}],
                }
                try:
                    body = self.bet(bd, access_token)
                    results.append(BetResult(
                        ok=True, race_id_vote=race_id_vote, horse_num=int(cand.horses[0]),
                        amount=int(cand.amount),
                        remaining_points=self._extract_remaining(body),
                        raw_response=body, verified=None,
                    ))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "%s の 1 買い目 失敗 (bet_id=%s money=%d): %s。他は続行",
                        race_id_vote, bet_id, int(cand.amount), e,
                    )
                    results.append(None)
        finally:
            self.logout(access_token)
        n_ok = sum(1 for r in results if r is not None)
        logger.info("multi bet split 完了: 成立 %d/%d (race=%s)",
                    n_ok, len(bets), race_id_vote)
        return results

    def place_win_bet(
        self,
        race_id_vote: str,
        horse_num: int,
        amount: int,
        deadline_ts: float | None = None,
        max_attempts: int = 2,
        retry_interval: float = 3.0,
        verify: bool = False,
    ) -> BetResult:
        """単勝 1 点投票 (ログイン → 投票 → ログアウトを一括実行)。

        Args:
            race_id_vote: 12 桁の netkeiba 形式 race_id (race_id_to_vote_id で変換)
            horse_num: 馬番 (1-18)
            amount: 投票額 (100 pt 単位)
            deadline_ts: 投票締切の UNIX 時刻。過ぎていたらリトライしない
                (締切 = 発走 3 分前。締切後の投票は無効なので粘らない)
            max_attempts: ログイン〜投票のリトライ回数 (一時的なネットワーク断対策)
            verify: True なら投票直後に GET /bet で反映を確認する。
                ただし仕様上「投票直後は非同期処理のため未反映になりうる」ため
                既定は False。反映確認は 60 秒以上あけて verify_many() で行う
                (run_live は次レースの処理時にまとめて確認する)。
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
