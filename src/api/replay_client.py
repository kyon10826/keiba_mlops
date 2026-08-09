"""過去の record_data から当日データ API のレスポンスを再構成するリプレイ用クライアント。

目的: 試験運用日 (8/15) まで大会 API は叩けないため、run_live.py の
「出走表取得 → 特徴量構築 → 予測 → レース毎のオッズ取得 → 投票判定」の
全経路を過去日でオフライン検証する。

制約 (実 API との差分):
  - オッズは「確定オッズ」を 5 分前オッズの代用にする (実運用より情報が新しい)
  - start_time は合成値 (record_data に発走時刻がないため)。--skip-wait 前提
  - runtable は結果列を持ったまま渡すが、特徴量は全て shift(1) ベースなので
    当該レース自身の結果は特徴量に混入しない (build_day_features 側の
    cutoff で履歴からも当日以降を除外する)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ERROR_CODES_EXCLUDE = [1, 3]  # 出走取消 / 競走除外


def _full_year(y: int) -> int:
    return 2000 + y if y < 100 else y


def date_str_of(df: pd.DataFrame) -> pd.Series:
    """record_data の year/month/day 列から 'YYYYMMDD' 文字列を作る。"""
    yyyy = df["year"].astype(int).apply(_full_year)
    return (
        yyyy.astype(str)
        + df["month"].astype(int).astype(str).str.zfill(2)
        + df["day"].astype(int).astype(str).str.zfill(2)
    )


class ReplayDataClient:
    """MastersDataClient と同じインターフェースで過去日を再生する。"""

    def __init__(self, hist: pd.DataFrame, date_str: str):
        """
        Args:
            hist: load_all_data で読み込んだ全期間の record_data (結合済み)
            date_str: リプレイ対象日 "YYYYMMDD" (test 期間内の開催日を指定)
        """
        self.date_str = date_str
        all_dates = date_str_of(hist)
        day = hist[all_dates == date_str].copy()
        if day.empty:
            available = sorted(all_dates.unique())
            near = [d for d in available if d[:6] == date_str[:6]][:10]
            raise SystemExit(
                f"リプレイ対象日 {date_str} のデータがありません。"
                f"同月の開催日: {near}"
            )
        # 出走取消・競走除外は実際の出走表に載らないので除外
        day = day[~day["error_code"].isin(ERROR_CODES_EXCLUDE)]
        self.day_df = day.reset_index(drop=True)
        logger.info(
            "[REPLAY] %s: %d レース %d 頭を再生",
            date_str, self.day_df.groupby(["place", "race_num"]).ngroups, len(self.day_df),
        )

    def get_racecards(self, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """timetable / runtable を record_data から再構成する。"""
        races = (
            self.day_df.groupby(["place", "race_num"], as_index=False)
            .size()
            .sort_values(["race_num", "place"])
            .reset_index(drop=True)
        )
        # 発走時刻は合成 (9:50 から 12 分間隔)。--skip-wait 前提なので順序だけ保つ
        races["start_time"] = [
            f"{9 + (50 + 12 * i) // 60:02d}:{(50 + 12 * i) % 60:02d}"
            for i in range(len(races))
        ]
        timetable = races[["place", "race_num", "start_time"]]
        runtable = self.day_df.copy()
        return timetable, runtable

    def get_win_odds(self, odds_race_id: str) -> pd.DataFrame:
        """確定オッズを 5 分前オッズの代用として返す (comb, odds)。"""
        from src.api.masters_client import race_id_to_odds_id

        day = self.day_df
        ids = day["race_id"].apply(race_id_to_odds_id)
        rows = day[ids == str(odds_race_id)]
        rows = rows[pd.to_numeric(rows["win_odds"], errors="coerce").fillna(0) > 0]
        return pd.DataFrame({
            "comb": rows["horse_num"].astype(int).astype(str).str.zfill(2),
            "odds": pd.to_numeric(rows["win_odds"], errors="coerce"),
        }).reset_index(drop=True)


def summarize_replay(bet_records: list[dict], day_df: pd.DataFrame) -> None:
    """リプレイ終了後、実結果 (rank / 確定オッズ) で損益を集計する。"""
    if not bet_records:
        print("[REPLAY] 投票判定なし")
        return
    bets = pd.DataFrame(bet_records)
    results = day_df[["place", "race_num", "horse_num", "rank", "win_odds"]].copy()
    results["horse_num"] = results["horse_num"].astype(int)
    merged = bets.merge(results, on=["place", "race_num", "horse_num"], how="left")

    merged["hit"] = (pd.to_numeric(merged["rank"], errors="coerce") == 1).astype(int)
    merged["payout"] = merged["hit"] * merged["amount"] * pd.to_numeric(
        merged["win_odds"], errors="coerce"
    ).fillna(0)

    wagered = int(merged["amount"].sum())
    payout = float(merged["payout"].sum())
    print("\n" + "=" * 60)
    print(f"[REPLAY 結果] {len(merged)} レース投票 / 的中 {int(merged['hit'].sum())}")
    print(f"  投票額: {wagered:,} pt / 払戻: {payout:,.0f} pt / "
          f"収支: {payout - wagered:+,.0f} pt (ROI {(payout / wagered - 1) * 100:+.1f}%)")
    for kind, seg in merged.groupby("bet_kind"):
        w = int(seg["amount"].sum())
        p = float(seg["payout"].sum())
        print(f"  [{kind}] {len(seg)} 件 投{w:,} 払{p:,.0f} ({(p / w - 1) * 100:+.1f}%)")
    print("=" * 60)
