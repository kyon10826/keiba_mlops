#!/usr/bin/env python3
"""AI 競馬予想マスターズ API 疎通確認スクリプト (試験運用日用)。

使い方:
    # データ API のみ確認 (認証情報なしで OK)
    python scripts/check_masters_api.py --date 20260815

    # 投票 API のログイン + 投票内容確認も行う (投票はしない)
    export NETKEIBA_LOGIN_ID="..."
    export NETKEIBA_PASSWORD="..."
    python scripts/check_masters_api.py --date 20260815 --check-vote

確認項目:
    [1] 当日データ API: Racecards が取得でき、timetable / runtable が非空か
    [2] 当日データ API: 最初のレースの Odds が取得できるか (単勝オッズの構造確認)
    [3] 投票 API: ログインしてトークンが取得できるか (--check-vote 時)
    [4] 投票 API: GET /bet で投票内容確認が通るか (--check-vote 時)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.masters_client import (
    MastersDataClient,
    MastersVoteClient,
    race_id_to_odds_id,
)


def main():
    parser = argparse.ArgumentParser(description="Masters API 疎通確認")
    parser.add_argument("--date", default=None, help="対象日 YYYYMMDD (省略時は今日)")
    parser.add_argument("--check-vote", action="store_true", help="投票 API のログイン確認も行う")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    ok = True

    # [1] Racecards
    print(f"[1] 当日データ API: Racecards ({date_str}) ...")
    client = MastersDataClient()
    try:
        timetable, runtable = client.get_racecards(date_str)
        print(f"    ✓ timetable {len(timetable)} レース / runtable {len(runtable)} 頭")
        if timetable.empty or runtable.empty:
            print("    ⚠️ 空データ。開催日か API 状態を確認")
            ok = False
        else:
            print(f"    timetable columns: {list(timetable.columns)}")
            print(f"    runtable columns ({len(runtable.columns)}): {list(runtable.columns)[:12]}...")
            print(timetable.head(3).to_string(index=False))
    except Exception as e:
        print(f"    ✗ 失敗: {e}")
        ok = False
        timetable, runtable = None, None

    # [2] Odds (最初のレース)
    if runtable is not None and not runtable.empty and "race_id" in runtable.columns:
        first_rid = runtable["race_id"].iloc[0]
        odds_id = race_id_to_odds_id(first_rid)
        print(f"\n[2] 当日データ API: Odds (race_id={first_rid} → odds_id={odds_id}) ...")
        try:
            win = client.get_win_odds(odds_id)
            if win.empty:
                print("    ⚠️ 単勝オッズが空 (発走 4分30秒前より前だと未提供の可能性)")
            else:
                print(f"    ✓ 単勝オッズ {len(win)} 頭分")
                print(win.head(5).to_string(index=False))
        except Exception as e:
            print(f"    ✗ 失敗: {e} (発走前でないと提供されない場合あり)")

    # [3][4] 投票 API
    if args.check_vote:
        login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
        password = os.environ.get("NETKEIBA_PASSWORD", "")
        if not login_id or not password:
            print("\n[3] 投票 API: NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD が未設定のためスキップ")
        else:
            print("\n[3] 投票 API: ログイン ...")
            vote = MastersVoteClient(login_id, password)
            try:
                token = vote.login()
                print(f"    ✓ アクセストークン取得 (先頭10文字: {token[:10]}...)")
                print("\n[4] 投票 API: 投票内容確認 (GET /bet) ...")
                try:
                    if runtable is not None and not runtable.empty:
                        from src.api.masters_client import race_id_to_vote_id
                        vote_id = race_id_to_vote_id(runtable["race_id"].iloc[0])
                        body = vote.check_bets([vote_id], token)
                        print(f"    ✓ 応答: {str(body)[:300]}")
                    else:
                        print("    (runtable が無いためスキップ)")
                except Exception as e:
                    print(f"    ✗ 失敗: {e}")
                vote.logout(token)
                print("    ✓ ログアウト")
            except Exception as e:
                print(f"    ✗ ログイン失敗: {e}")
                ok = False

    print("\n" + ("✅ 疎通確認 OK" if ok else "⚠️ 一部失敗あり — 上のログを確認"))


if __name__ == "__main__":
    main()
