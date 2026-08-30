#!/usr/bin/env python3
"""投票 API の単発疎通テスト (100pt 単勝 1 レースだけ)。

試験運用日に「投票 API が実際に動くか」を最小侵襲で確認する用途。
run_live.py と違い全レース回さず、指定した 1 レース・1 馬番だけを叩く。

使い方:
    export NETKEIBA_LOGIN_ID="..." NETKEIBA_PASSWORD="..."

    # 1) まずログインだけ試す (投票しない、認証情報の確認用)
    python scripts/vote_smoke.py --login-only

    # 2) 発走時刻を目安に、指定レースの top-1 予測馬に 100pt 投票
    python scripts/vote_smoke.py --date 20260823 --place 中京 --race-num 12

    # 3) 手動で馬番を指定して 100pt 投票
    python scripts/vote_smoke.py --race-id-vote 202608120612 --horse-num 5

    # 4) 投票後 60 秒待って GET /bet で反映確認 (--verify を付ける)
    python scripts/vote_smoke.py --date 20260823 --place 中京 --race-num 12 --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.api.masters_client import (
    MastersDataClient, MastersVoteClient,
    normalize_runtable, race_id_to_odds_id, race_id_to_vote_id,
)


def resolve_top1(cfg_path: str, date: str, place: str, race_num: int) -> tuple[str, int]:
    """指定レースの top-1 予測馬を返す (models_masters + 履歴 + netkeiba 補完)。"""
    from src.data.loader import load_config
    _SCR = os.path.dirname(os.path.abspath(__file__))
    if _SCR not in sys.path:
        sys.path.insert(0, _SCR)
    from run_live import (
        build_day_features, predict_win_probs, enrich_from_netkeiba,
        impute_weight_from_history,
    )
    from src.data.loader import load_all_data
    cfg = load_config(cfg_path)
    model_dir = cfg["model"]["dir"]

    c = MastersDataClient()
    timetable, runtable = c.get_racecards(date)
    runtable = normalize_runtable(runtable, timetable)
    runtable = enrich_from_netkeiba(runtable)
    train_df, valid_df, test_df = load_all_data(cfg)
    hist = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    runtable = impute_weight_from_history(runtable, hist)

    day_feat = build_day_features(cfg, runtable, model_dir, date, hist=hist)
    day_feat = predict_win_probs(day_feat, model_dir)

    rows = day_feat[(day_feat["place"] == place) & (day_feat["race_num"] == int(race_num))]
    if rows.empty:
        raise SystemExit(f"該当レースなし: {place} {race_num}R")
    top = rows.sort_values("win_pred_prob", ascending=False).iloc[0]
    rid_vote = race_id_to_vote_id(int(top["race_id"]))
    return rid_vote, int(top["horse_num"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/masters_2026.yaml")
    p.add_argument("--login-only", action="store_true", help="投票せずログインだけ試す")
    p.add_argument("--date", default=None, help="対象日 YYYYMMDD")
    p.add_argument("--place", default=None)
    p.add_argument("--race-num", type=int, default=None)
    p.add_argument("--race-id-vote", default=None, help="12 桁の投票用 race_id (直接指定)")
    p.add_argument("--horse-num", type=int, default=None, help="馬番 (直接指定)")
    p.add_argument("--amount", type=int, default=100, help="投票額 (100 pt 単位、既定 100)")
    p.add_argument("--verify", action="store_true", help="投票 60 秒後に GET /bet で反映確認")
    args = p.parse_args()

    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password = os.environ.get("NETKEIBA_PASSWORD", "")
    if not login_id or not password:
        raise SystemExit("NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD を設定してください")

    client = MastersVoteClient(login_id, password)

    if args.login_only:
        print("[1] ログイン試行 ...")
        token = client.login()
        print(f"    ✓ access_token={token[:12]}... (先頭のみ)")
        client.logout(token)
        print("    ✓ ログアウトも成功")
        return

    if args.race_id_vote and args.horse_num:
        rid_vote = args.race_id_vote
        horse_num = args.horse_num
        print(f"[指定] race_id_vote={rid_vote} 馬番{horse_num} {args.amount}pt")
    elif args.date and args.place and args.race_num:
        print(f"[予測] {args.date} {args.place} {args.race_num}R の top-1 を算出 ...")
        rid_vote, horse_num = resolve_top1(args.config, args.date, args.place, args.race_num)
        print(f"    → race_id_vote={rid_vote} 馬番{horse_num} {args.amount}pt")
    else:
        raise SystemExit("--login-only か、(--date --place --race-num) か、(--race-id-vote --horse-num) を指定")

    print(f"[投票] race={rid_vote} 馬番{horse_num} {args.amount}pt を投票します ...")
    result = client.place_win_bet(rid_vote, horse_num, args.amount)
    print(f"    ✓ ok={result.ok} remaining={result.remaining_points}")
    print(f"    raw_response: {result.raw_response}")

    if args.verify:
        print("[確認] 60 秒待機してから GET /bet で反映確認 ...")
        time.sleep(60)
        r = client.verify_many([(rid_vote, horse_num, args.amount)])
        v = r.get(str(rid_vote))
        if v is True:
            print(f"    ✓ プラットフォーム反映 OK (race={rid_vote} 馬番{horse_num} {args.amount}pt)")
        elif v is False:
            print(f"    ✗ 未反映または不一致。netkeiba マイページで目視確認してください")
        else:
            print(f"    ? 判定不能 (レスポンス構造が想定外)")


if __name__ == "__main__":
    main()
