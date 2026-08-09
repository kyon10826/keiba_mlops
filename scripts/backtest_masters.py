#!/usr/bin/env python3
"""AI 競馬予想マスターズ バックテスト実行スクリプト。

使い方:
    python scripts/backtest_masters.py --config config/backtest_masters_prod.yaml

流れ:
    1. 学習済みの複勝モデル + 単勝モデル + パイプラインをロード
    2. バックテスト対象期間 (config の backtest.start_date/end_date) のレースを抽出
    3. 特徴量パイプラインを適用 → 単勝モデルで予測 → Isotonic 校正
    4. コンペルール準拠のシミュレータで投票戦略を回す
    5. 結果を出力 (ROI, 的中率, 96レース×50万pt制約充足, 日次)

【レビュー時の注意】
- config の data セクションは学習時 (masters_2026.yaml) と完全一致させること。
  train/valid/test の行数が変わると iloc 分割の位置がズレて test が壊れる。
- ここで使う win_odds は「確定オッズ」。実運用の「5 分前オッズ」とは
  ズレるため、バックテスト成績はやや楽観的に出る (masters_simulator.py 冒頭参照)。
- 戦略パラメータの網羅的な比較はこのスクリプトではなく
  dump_predictions.py + tune_strategy.py の高速ループで行う (10 分 → 数秒)。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config
from src.features.pipeline import (
    FeaturePipeline,
    FEATURE_COLUMNS,
    assert_no_market_info,
)
from src.model.trainer import load_model
from src.model.calibrator import HoldoutCalibrator
from src.model.win_trainer import load_win_model, load_win_meta
from src.strategy.masters_simulator import (
    print_daily_summary,
    print_masters_summary,
    print_top_bets,
    simulate_masters_backtest,
)


def main():
    parser = argparse.ArgumentParser(description="AI 競馬予想マスターズ バックテスト")
    parser.add_argument("--config", default="config/backtest_masters.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    model_dir = cfg["model"]["dir"]

    # Step 1: モデルロード
    print("=" * 70)
    print("Step 1: Loading models...")
    print("=" * 70)
    show_model = load_model(os.path.join(model_dir, "lgbm_model.txt"))
    calibrator = HoldoutCalibrator.load(os.path.join(model_dir, "calibrator.pkl"))
    win_model = load_win_model(os.path.join(model_dir, "win_model.txt"))
    win_meta = load_win_meta(os.path.join(model_dir, "win_meta.pkl"))
    print(f"  複勝モデル: {model_dir}/lgbm_model.txt")
    print(f"  単勝モデル: {model_dir}/win_model.txt (feature_columns: {len(win_meta['feature_columns'])} 列)")

    # Step 2: データ読み込み + 特徴量パイプライン
    print("\n" + "=" * 70)
    print("Step 2: Loading data + feature engineering...")
    print("=" * 70)
    train_df, valid_df, test_df = load_all_data(cfg)
    print(f"  Train: {len(train_df)} / Valid: {len(valid_df)} / Test: {len(test_df)}")

    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    try:
        pipeline = FeaturePipeline.load(pipeline_path, cfg)
        print("  Pipeline loaded from pickle.")
    except Exception as e:
        print(f"  Pipeline load failed ({e.__class__.__name__}), rebuilding...")
        pipeline = FeaturePipeline(cfg)
        pipeline.fit(train_df)

    all_data = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    all_transformed = pipeline.transform(all_data)

    n_train = len(train_df)
    n_valid = len(valid_df)
    test_feat = all_transformed.iloc[n_train + n_valid:].reset_index(drop=True)

    # 単勝モデルに使う特徴量 (学習時と同じ列を使用)
    available_features = [c for c in FEATURE_COLUMNS if c in test_feat.columns]
    assert_no_market_info(available_features)
    print(f"  Features: {len(available_features)} columns (オッズ非依存)")

    # Step 3: 予測 (単勝モデル + キャリブレーション)
    print("\n" + "=" * 70)
    print("Step 3: Predicting with WIN model...")
    print("=" * 70)
    for c in win_meta["feature_columns"]:
        if c not in test_feat.columns:
            test_feat[c] = 0.0
    raw_win = win_model.predict(test_feat[win_meta["feature_columns"]])

    win_cal_path = os.path.join(model_dir, "win_calibrator.pkl")
    if os.path.exists(win_cal_path):
        win_calibrator = HoldoutCalibrator.load(win_cal_path)
        test_feat["win_pred_prob"] = win_calibrator.predict(raw_win)
        print(f"  Win calibrator applied: prob range "
              f"[{test_feat['win_pred_prob'].min():.4f}, {test_feat['win_pred_prob'].max():.4f}]")
    else:
        test_feat["win_pred_prob"] = raw_win
        print("  No win calibrator found; using raw probs")

    # 参考として複勝モデルの予測も付与
    raw = show_model.predict(test_feat[available_features])
    test_feat["show_pred_prob"] = calibrator.predict(raw)

    print(f"  win_pred_prob: mean={test_feat['win_pred_prob'].mean():.4f}, "
          f"max={test_feat['win_pred_prob'].max():.4f}")

    # Step 4: シミュレーション実行
    print("\n" + "=" * 70)
    print("Step 4: Running Masters backtest simulation...")
    print("=" * 70)
    bt = cfg.get("backtest", {})
    print(f"  期間: {bt.get('start_date', '(all)')} 〜 {bt.get('end_date', '(all)')}")
    print(f"  投票戦略: {cfg['strategy'].get('bet_type', 'win')} "
          f"(win_prob_min={cfg['strategy'].get('win_prob_min')}, "
          f"bet_amount={cfg['strategy'].get('win_bet_amount')} pt)")

    result = simulate_masters_backtest(test_feat, cfg)

    # Step 5: サマリ出力
    print()
    print_masters_summary(result)
    print_daily_summary(result)
    print_top_bets(result, n=10)

    # 詳細 CSV に保存
    out_dir = model_dir
    if not result.bets.empty:
        bets_csv = os.path.join(out_dir, "masters_backtest_bets.csv")
        result.bets.to_csv(bets_csv, index=False)
        print(f"\n  投票詳細: {bets_csv}")
    if not result.daily_summary.empty:
        daily_csv = os.path.join(out_dir, "masters_backtest_daily.csv")
        result.daily_summary.to_csv(daily_csv, index=False)
        print(f"  日次サマリ: {daily_csv}")


if __name__ == "__main__":
    main()
