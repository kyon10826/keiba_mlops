#!/usr/bin/env python3
"""投票ログとプラットフォーム登録内容の照合 (試験運用日・本番日の締め作業)。

大会ルール「API のステータスが正常でも、プラットフォームへの反映がされていない
場合、その投票は無効」への防御として、run_live.py が書いた投票ログ CSV と
投票 API (GET /bet) の登録内容を突き合わせる。

使い方:
    export NETKEIBA_LOGIN_ID="..."
    export NETKEIBA_PASSWORD="..."
    python scripts/verify_bets.py --date 20260815

出力:
    レースごとに OK / MISSING / MISMATCH を表示。
    - OK       = 馬番・金額ともログと一致
    - MISMATCH = レースは登録済みだが馬番か金額が異なる
    - MISSING  = プラットフォームに登録が見つからない (→ 目視確認 + 運営報告)
    1 件でも OK 以外があれば終了コード 1。

注意: トークンは 5 分で失効するため、レース数が多い日もログイン 1 回で
      まとめて確認できるようチャンク処理している (GET /bet は複数 race_id 可)。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.api.masters_client import MastersVoteClient, extract_registered_bets

CHUNK_SIZE = 12  # 1 リクエストで確認する race_id 数


def compare_row(row: pd.Series, registered: dict[str, list[dict]]) -> str:
    """ログ 1 行と登録内容を比較して OK / MISSING / MISMATCH を返す。"""
    rid = str(row["race_id_vote"])
    bets = registered.get(rid, [])
    if not bets:
        return "MISSING"
    expected_bet_id = f"b1_c0_{int(row['horse_num'])}"
    for b in bets:
        if str(b.get("bet_id")) == expected_bet_id:
            money = pd.to_numeric(b.get("money"), errors="coerce")
            if pd.notna(money) and int(money) == int(row["amount"]):
                return "OK"
            return "MISMATCH"
    return "MISMATCH"


def main():
    parser = argparse.ArgumentParser(description="投票ログとプラットフォームの照合")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--log", default=None, help="投票ログ CSV (省略時は date から自動)")
    args = parser.parse_args()

    log_path = args.log or f"logs/masters_live_bets_{args.date}.csv"
    if not os.path.exists(log_path):
        raise SystemExit(f"投票ログが見つかりません: {log_path}")

    df = pd.read_csv(log_path)
    if "dry_run" in df.columns:
        df = df[df["dry_run"] != True]  # noqa: E712
    if df.empty:
        print(f"{log_path}: 本番投票の記録がありません (dry-run のみ)")
        return

    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password = os.environ.get("NETKEIBA_PASSWORD", "")
    if not login_id or not password:
        raise SystemExit("NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD を設定してください")

    client = MastersVoteClient(login_id, password)
    race_ids = df["race_id_vote"].astype(str).unique().tolist()

    registered: dict[str, list[dict]] = {}
    token = client.login()
    try:
        for i in range(0, len(race_ids), CHUNK_SIZE):
            chunk = race_ids[i:i + CHUNK_SIZE]
            body = client.check_bets(chunk, token)
            registered.update(extract_registered_bets(body))
    finally:
        client.logout(token)

    print(f"\n=== 投票照合 {args.date} ({len(df)} 件) ===")
    print(f"{'場所':<6}{'R':>3} {'race_id':<14}{'馬番':>4}{'金額':>8}  結果")
    n_bad = 0
    for _, row in df.iterrows():
        status = compare_row(row, registered)
        if status != "OK":
            n_bad += 1
        print(f"{str(row.get('place', '?')):<6}{int(row['race_num']):>3} "
              f"{str(row['race_id_vote']):<14}{int(row['horse_num']):>4}"
              f"{int(row['amount']):>8}  {status}")

    if n_bad:
        print(f"\n★ {n_bad} 件が未反映または不一致です。netkeiba マイページで目視確認し、"
              "未反映なら運営 Slack に報告してください。")
        sys.exit(1)
    print("\n全件一致。プラットフォーム反映を確認しました。")


if __name__ == "__main__":
    main()
