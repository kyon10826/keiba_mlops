#!/usr/bin/env python3
"""ハイブリッド投票戦略のグリッドサーチ (予測ダンプ上で瞬時に評価)。

前提: scripts/dump_predictions.py で test_predictions.csv を作成済みであること。

================================================================================
【レビューガイド】評価方法論
================================================================================
■ なぜダンプ + グリッドサーチに分けるか
  戦略パラメータ (EV 閾値・オッズ帯) はモデルと独立なので、
  「予測確率を一度だけ計算して保存 → 戦略評価は保存済み CSV 上で行う」形にすると、
  特徴量再構築 (~10 分) を 1 回で済ませ、384 設定を数秒で評価できる。
  ML の実験管理でいう「予測とデシジョンの分離」。

■ 過剰適合の防御: 窓を分けたチューニング/検証
  - チューニング窓 (2024-08/09): グリッドサーチでベスト設定を探す
  - 検証窓 (2025-08/09): ベスト設定を「そのまま」適用して成績を見る
  384 設定も試せば 2024 窓ではどれかが偶然良く見える (multiple comparisons)。
  2025 窓でも勝つ設定だけが本物 — という walk-forward 的な検証構造。

■ この分析から得られた結論 (2026-08 時点)
  両窓でベースライン (sub のみ = 全レース top-1 少額) を上回る main 設定はゼロ。
  → オッズなしモデルが市場と強く乖離した馬 (高予測確率 × 中〜高オッズ) は、
    市場の方が正しいことが多い (adverse selection)。
  → 本番戦略は EV フィルタを捨て、favorite_concentration (run_live.py) を採用。

■ n_main < 5 の設定を捨てる理由
  main が 4 回以下だと 1 発の的中/不的中で ROI が ±50% 動き、評価にならない。
================================================================================

使い方:
    python scripts/tune_strategy.py --pred models_masters/test_predictions.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

ERROR_CODES_EXCLUDE = [1, 3]

TUNE_WINDOW = ("2024-08-24", "2024-09-22")
VALID_WINDOW = ("2025-08-30", "2025-09-21")

MAIN_AMOUNT = 3000
SUB_AMOUNT = 2000
INITIAL = 1_000_000

# グリッド
GRID = {
    "win_prob_min": [0.10, 0.12, 0.15],
    "min_ev": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0],
    "min_odds": [3.0, 5.0, 7.0, 10.0],
    "max_odds": [15.0, 20.0, 30.0, 50.0],
}


def build_race_top1(pred_df: pd.DataFrame) -> pd.DataFrame:
    """レース毎の top-1 (win_pred_prob 最大、除外馬を除く) テーブルを作る。"""
    df = pred_df.copy()
    df = df[~df["error_code"].isin(ERROR_CODES_EXCLUDE)]
    df = df[pd.to_numeric(df["win_odds"], errors="coerce").fillna(0) > 0]

    df["year_full"] = df["year"].astype(int).apply(lambda y: 2000 + y if y < 100 else y)
    df["date"] = pd.to_datetime(
        df[["year_full", "month", "day"]].rename(columns={"year_full": "year"}),
        errors="coerce",
    )

    race_key = ["date", "place", "race_num"]
    idx = df.groupby(race_key)["win_pred_prob"].idxmax()
    top1 = df.loc[idx, ["date", "place", "race_num", "horse_num", "rank",
                        "win_odds", "win_pred_prob"]].reset_index(drop=True)
    top1["hit"] = (top1["rank"] == 1).astype(int)
    return top1


def evaluate(top1: pd.DataFrame, cfg: dict) -> dict:
    """1 設定のハイブリッド戦略を top-1 テーブル上で評価する。"""
    prob = top1["win_pred_prob"].values
    odds = top1["win_odds"].values
    hit = top1["hit"].values

    is_main = (
        (prob >= cfg["win_prob_min"])
        & (odds >= cfg["min_odds"])
        & (odds <= cfg["max_odds"])
        & (prob * odds >= cfg["min_ev"])
    )
    amount = np.where(is_main, MAIN_AMOUNT, SUB_AMOUNT).astype(float)
    payout = amount * odds * hit

    n_main = int(is_main.sum())
    main_hits = int(hit[is_main].sum())
    main_wager = float(amount[is_main].sum())
    main_payout = float(payout[is_main].sum())

    total_wager = float(amount.sum())
    total_payout = float(payout.sum())

    return {
        "n_races": len(top1),
        "n_main": n_main,
        "main_hits": main_hits,
        "main_roi": (main_payout - main_wager) / main_wager if main_wager > 0 else 0.0,
        "total_wagered": total_wager,
        "roi": (total_payout - total_wager) / total_wager if total_wager > 0 else 0.0,
        "final": INITIAL + total_payout - total_wager,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="models_masters/test_predictions.csv")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred, low_memory=False)
    top1 = build_race_top1(pred_df)
    print(f"top-1 テーブル: {len(top1)} レース "
          f"({top1['date'].min().date()} 〜 {top1['date'].max().date()})")

    tune = top1[(top1["date"] >= TUNE_WINDOW[0]) & (top1["date"] <= TUNE_WINDOW[1])]
    valid = top1[(top1["date"] >= VALID_WINDOW[0]) & (top1["date"] <= VALID_WINDOW[1])]
    print(f"チューニング窓 (2024): {len(tune)} レース / 検証窓 (2025): {len(valid)} レース")

    # ベースライン: 全レース sub のみ (main なし)
    base_cfg = {"win_prob_min": 99, "min_ev": 99, "min_odds": 0, "max_odds": 0}
    b_t, b_v = evaluate(tune, base_cfg), evaluate(valid, base_cfg)
    print(f"\n[ベースライン: sub のみ] 2024: 最終 {b_t['final']:,.0f} (ROI {b_t['roi']*100:+.1f}%)"
          f" / 2025: 最終 {b_v['final']:,.0f} (ROI {b_v['roi']*100:+.1f}%)")

    # グリッドサーチ
    rows = []
    keys = list(GRID.keys())
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, values))
        if cfg["min_odds"] >= cfg["max_odds"]:
            continue
        r_tune = evaluate(tune, cfg)
        if r_tune["n_main"] < 5:  # main が少なすぎる設定はスキップ (統計的に無意味)
            continue
        r_valid = evaluate(valid, cfg)
        rows.append({
            **cfg,
            "tune_n_main": r_tune["n_main"],
            "tune_main_hits": r_tune["main_hits"],
            "tune_main_roi": r_tune["main_roi"],
            "tune_final": r_tune["final"],
            "valid_n_main": r_valid["n_main"],
            "valid_main_hits": r_valid["main_hits"],
            "valid_main_roi": r_valid["main_roi"],
            "valid_final": r_valid["final"],
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("有効な設定なし")
        return

    # 2024 最終ポイント順で上位を表示し、2025 での成績も併記
    res = res.sort_values("tune_final", ascending=False)
    print(f"\n=== 2024 窓ベスト {args.top_k} 設定 (右 2 列が 2025 検証) ===")
    show_cols = ["win_prob_min", "min_ev", "min_odds", "max_odds",
                 "tune_n_main", "tune_main_hits", "tune_main_roi", "tune_final",
                 "valid_n_main", "valid_main_hits", "valid_main_roi", "valid_final"]
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(res[show_cols].head(args.top_k).to_string(index=False))

    # 両窓で頑健な設定 (tune と valid 両方でベースライン超え)
    robust = res[
        (res["tune_final"] > b_t["final"]) & (res["valid_final"] > b_v["final"])
    ].copy()
    if robust.empty:
        print("\n⚠️ 両窓でベースライン (sub のみ) を上回る main 設定は無し")
        print("   → main ベットに頑健なエッジが無い。sub 中心 + main は極少数に絞るべき")
    else:
        robust["combined"] = robust["tune_final"] + robust["valid_final"]
        robust = robust.sort_values("combined", ascending=False)
        print(f"\n=== 両窓でベースライン超えの頑健設定 ({len(robust)} 件) ===")
        with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
            print(robust[show_cols].head(args.top_k).to_string(index=False))

    out = "models_masters/strategy_grid_results.csv"
    res.to_csv(out, index=False)
    print(f"\n全 {len(res)} 設定の結果: {out}")


if __name__ == "__main__":
    main()
