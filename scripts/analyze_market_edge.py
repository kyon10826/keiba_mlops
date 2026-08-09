#!/usr/bin/env python3
"""市場残差モデルの EV バケット分析 — 実マネー戦略の採否を決める中核証拠。

前提: dump_predictions.py で win_market_pred_prob 入りのダンプを作成済みであること。

================================================================================
【レビューガイド】何を検証するか
================================================================================
市場残差モデルの主張は「校正後確率 p が市場確率より正確」であること。
これが本当なら、EV = p × オッズ が高い馬ほど実現 ROI が高くなるはず。

検証方法 (EV バケット分析):
  全出走馬 (top-1 に限らない) を EV = p×odds の値でバケットに分け、
  各バケットの「フラット 100pt 投票の実現 ROI」を計測する。
    - EV < 0.8 のバケット: 大きくマイナスのはず (賭ないべき馬)
    - EV ≥ 1.0 のバケット: ここがプラスなら実マネーで賭ける根拠になる
  さらに 2024 窓 (チューニング) と 2025 窓 (検証) で別々に見て、
  再現性を確認する (multiple comparisons への防御)。

判定基準:
  「EV ≥ 1.05 バケットの実現 ROI が両窓でプラス」なら ev_market 戦略を採用。
  片窓のみプラスなら保留 (サンプル不足)、両窓マイナスなら市場残差でも
  エッジなしと結論して favorite_concentration を維持する。
================================================================================

使い方:
    python scripts/analyze_market_edge.py --pred models_masters/test_predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

ERROR_CODES_EXCLUDE = [1, 3]

WINDOWS = {
    "全 test 期間": (None, None),
    "2024 窓 (チューニング)": ("2024-08-24", "2024-09-22"),
    "2025 窓 (検証)": ("2025-08-30", "2025-09-21"),
}

EV_BUCKETS = [0.0, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 99.0]


def load_pred(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "win_market_pred_prob" not in df.columns:
        raise SystemExit("win_market_pred_prob 列がありません。market ヘッド学習後に "
                         "dump_predictions.py を再実行してください。")
    df = df[~df["error_code"].isin(ERROR_CODES_EXCLUDE)]
    df = df[pd.to_numeric(df["win_odds"], errors="coerce").fillna(0) > 0].copy()
    df["year_full"] = df["year"].astype(int).apply(lambda y: 2000 + y if y < 100 else y)
    df["date"] = pd.to_datetime(
        df[["year_full", "month", "day"]].rename(columns={"year_full": "year"}),
        errors="coerce",
    )
    df["hit"] = (df["rank"] == 1).astype(int)
    df["ev"] = df["win_market_pred_prob"] * df["win_odds"]
    return df


def bucket_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ({len(df)} 頭 / {df.groupby(['date','place','race_num']).ngroups} レース) ===")
    print(f"{'EVバケット':<14}{'頭数':>7}{'的中率':>8}{'平均odds':>9}{'実現ROI':>10}")
    for lo, hi in zip(EV_BUCKETS[:-1], EV_BUCKETS[1:]):
        seg = df[(df["ev"] >= lo) & (df["ev"] < hi)]
        if len(seg) == 0:
            continue
        roi = (seg["hit"] * seg["win_odds"]).mean() - 1
        print(f"{lo:>5.2f}-{hi:<7.2f}{len(seg):>7}{seg['hit'].mean()*100:>7.1f}%"
              f"{seg['win_odds'].mean():>9.1f}{roi*100:>+9.1f}%")


def strategy_sim(df: pd.DataFrame, label: str, min_ev: float, flat: int = 1000,
                 max_odds: float | None = None) -> None:
    """EV ≥ min_ev の馬にフラット投票した場合の成績 (1 レース複数頭も許容)。

    max_odds を指定すると高オッズ帯を除外する。
    (EV 乖離は「特徴量が最も無力な超大穴」に集中しがちなので、
     中人気帯に限定した残差エッジの有無を切り分けるために使う)
    """
    bets = df[df["ev"] >= min_ev]
    if max_odds is not None:
        bets = bets[bets["win_odds"] <= max_odds]
    cap = f" & odds≤{max_odds:.0f}" if max_odds else ""
    if bets.empty:
        print(f"  [{label}] EV≥{min_ev}{cap}: 該当なし")
        return
    n_races = bets.groupby(["date", "place", "race_num"]).ngroups
    wag = flat * len(bets)
    pay = float((bets["hit"] * bets["win_odds"]).sum() * flat)
    print(f"  [{label}] EV≥{min_ev}{cap}: {len(bets)}頭/{n_races}R 的中{int(bets['hit'].sum())} "
          f"投{wag:,} 払{pay:,.0f} ROI {(pay-wag)/wag*100:+.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="models_masters/test_predictions.csv")
    args = parser.parse_args()

    df = load_pred(args.pred)

    # モデル確率 vs 市場確率の全体比較 (Brier)
    from sklearn.metrics import brier_score_loss
    b_model = brier_score_loss(df["hit"], df["win_market_pred_prob"].clip(0, 1))
    b_market = brier_score_loss(df["hit"], df["market_prob"].clip(0, 1))
    print(f"Brier 全体: model={b_model:.5f} vs market={b_market:.5f} "
          f"({'✓ モデルが市場より正確' if b_model < b_market else '✗ 市場に勝てていない'})")

    for label, (start, end) in WINDOWS.items():
        sub = df
        if start:
            sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
        bucket_table(sub, label)

    print("\n--- EV 閾値別のフラット投票シミュレーション ---")
    for label, (start, end) in WINDOWS.items():
        sub = df
        if start:
            sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
        for min_ev in [1.0, 1.05, 1.1, 1.2]:
            strategy_sim(sub, label, min_ev)

    print("\n--- 同上・オッズ上限つき (超大穴を除外した中人気帯の残差エッジ確認) ---")
    for label, (start, end) in WINDOWS.items():
        sub = df
        if start:
            sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
        for max_odds in [15.0, 30.0]:
            for min_ev in [1.0, 1.05]:
                strategy_sim(sub, label, min_ev, max_odds=max_odds)


if __name__ == "__main__":
    main()
