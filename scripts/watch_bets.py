#!/usr/bin/env python3
"""投票状況モニター (レースごとの賭け方を人間可読な表で表示)。

使い方:
    python scripts/watch_bets.py                    # 今日の全レース
    python scripts/watch_bets.py --tail 15          # 直近 15 レースのみ
    watch -n 15 python scripts/watch_bets.py        # 15 秒ごとに自動更新
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    p.add_argument("--tail", type=int, default=0, help="末尾 N レースだけ (0=全部)")
    p.add_argument("--dryrun", action="store_true", help="dry-run ログを見る")
    args = p.parse_args()

    suffix = "_dryrun" if args.dryrun else ""
    bets_path = f"logs/masters_live_bets_{args.date}{suffix}.csv"
    state_path = f"logs/masters_live_state{suffix}.json"

    print(f"\n=== 投票状況 {args.date} ({datetime.now().strftime('%H:%M:%S')}) ===\n")

    if not os.path.exists(bets_path):
        print(f"[未起動] 投票ログなし: {bets_path}")
        print("run_live.py の朝の予測フェーズ (起動から約 20 分) が終わり、\n"
              "最初のレースの発走 4 分 10 秒前を過ぎるまで何も書かれません。")
        return

    df = pd.read_csv(bets_path)
    state = {"races_bet": [], "total_wagered": 0, "remaining_points": None}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)

    if len(df) == 0:
        print("(まだ投票なし。1 レース目 = 9:40 発走の 4 分 10 秒前 = 9:35:50 頃から書かれ始めます)")
        return

    show = df.tail(args.tail) if args.tail else df

    KIND = {
        "fav_big":  "★大口",
        "min":      " 最低",
        "sub":      " 通常",
        "main":     " 本命",
        "ev":       "EV攻",
        "anaba":    "🎲穴狙",
        "trio":     "🎯三複",
        "trifecta": "💎三単",
        "quinella": " 馬連",
        "exacta":   " 馬単",
    }

    print(f"{'発走':>5}  {'場所':<3}{'R':>3}  {'種別':<5} {'馬番':>3} {'金額':>7}  {'勝率':>6} {'ODDS':>6} {'EV':>5}  反映")
    print("-" * 78)
    for _, r in show.iterrows():
        kind = KIND.get(str(r.get("bet_kind", "")), str(r.get("bet_kind", "?")))
        prob = f"{r['pred_prob'] * 100:.1f}%" if pd.notna(r["pred_prob"]) else "  --"
        odds = f"{r['odds']:.1f}" if pd.notna(r["odds"]) else "  --"
        ev = f"{r['ev']:.2f}" if pd.notna(r["ev"]) else " --"
        v = r.get("verified")
        if v is True or str(v).lower() == "true":
            ver = "  ✓"
        elif v is False or str(v).lower() == "false":
            ver = "  ✗"
        else:
            ver = "  ?"
        print(f"{r['start_time']:>5}  {r['place']:<3}{int(r['race_num']):>3}  {kind:<5} "
              f"{int(r['horse_num']):>3} {int(r['amount']):>7,}  {prob:>6} {odds:>6} {ev:>5}  {ver}")

    # サマリ
    print("-" * 78)
    total_bet = int(df["amount"].sum())
    n_races = len(df)
    n_big = int((df["bet_kind"] == "fav_big").sum())
    n_ok = int(df["verified"].astype(str).str.lower().eq("true").sum()) if "verified" in df.columns else 0
    n_bad = int(df["verified"].astype(str).str.lower().eq("false").sum()) if "verified" in df.columns else 0

    print(f"本日: {n_races} 件 (大口 {n_big})  投票額 {total_bet:,} pt  反映 ✓{n_ok} ✗{n_bad}")
    total_wag = state["total_wagered"]
    n_bet = len(state["races_bet"])
    rem = state.get("remaining_points")
    print(f"累積: {n_bet}/96 レース  累計 {total_wag:,}/500,000 pt  残 {rem if rem is not None else '?'} pt")
    r_pct = min(100.0, n_bet / 96 * 100)
    w_pct = min(100.0, total_wag / 500_000 * 100)
    print(f"      レース [{'█' * int(r_pct / 5):<20}] {r_pct:5.1f}%")
    print(f"      投票額 [{'█' * int(w_pct / 5):<20}] {w_pct:5.1f}%")

    if n_bad:
        print("\n★ 未反映投票あり: netkeiba マイページで目視確認 → 締切内なら再投票を検討")


if __name__ == "__main__":
    main()
