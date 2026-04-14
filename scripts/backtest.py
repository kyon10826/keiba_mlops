#!/usr/bin/env python3
"""バックテストのエントリーポイント。

使い方:
    python scripts/backtest.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config
from src.features.pipeline import FeaturePipeline, FEATURE_COLUMNS
from src.model.trainer import load_model
from src.model.calibrator import HoldoutCalibrator
from src.strategy.selector import select_bets_for_all_races, select_bets_dispatch
from src.strategy.simulator import simulate_backtest, print_backtest_summary, plot_bankroll_history


def main():
    parser = argparse.ArgumentParser(description="Backtest betting strategy")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--data-dir", default=None, help="Override data directory from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    model_dir = cfg["model"]["dir"]

    # モデルを読み込む
    print("Loading model...")
    model = load_model(os.path.join(model_dir, "lgbm_model.txt"))
    calibrator = HoldoutCalibrator.load(os.path.join(model_dir, "calibrator.pkl"))

    # データを読み込む
    print("Loading data...")
    train_df, valid_df, test_df = load_all_data(cfg)

    # パイプラインを読み込む、もしくは再構築
    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    try:
        pipeline = FeaturePipeline.load(pipeline_path, cfg)
        print("Pipeline loaded from pickle.")
    except (NotImplementedError, Exception) as e:
        print(f"Pipeline pickle load failed ({e.__class__.__name__}), rebuilding from training data...")
        pipeline = FeaturePipeline(cfg)
        pipeline.fit(train_df)

    # バックテストにはテストデータを使用
    all_data = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    all_transformed = pipeline.transform(all_data)

    n_train = len(train_df)
    n_valid = len(valid_df)
    test_feat = all_transformed.iloc[n_train + n_valid:].reset_index(drop=True)

    available_features = [c for c in FEATURE_COLUMNS if c in test_feat.columns]

    # 予測
    print("Predicting...")
    raw_probs = model.predict(test_feat[available_features])
    test_feat["pred_prob"] = calibrator.predict(raw_probs)

    # 馬券対象を選定
    print("Selecting bets...")
    strat = cfg["strategy"]
    selection_method = strat.get("selection_method", "threshold")
    if selection_method == "threshold":
        bet_df = select_bets_dispatch(
            test_feat,
            method="threshold",
            prob_threshold=strat.get("prob_threshold", 0.3),
            max_popularity=strat.get("max_popularity", 3),
        )
    else:
        bet_df = select_bets_dispatch(
            test_feat,
            method="ev",
            top_n_popularity=strat.get("top_n_popularity", 3),
            min_expected_value=strat.get("min_expected_value", 1.0),
        )

    if bet_df.empty:
        print("No bets selected!")
        return

    print(f"Selected {len(bet_df)} bets out of "
          f"{test_feat.groupby(['year', 'month', 'day', 'place', 'race_num']).ngroups} races")

    # シミュレーション実行
    print("\nRunning backtest simulation...")
    results = simulate_backtest(bet_df, cfg)
    print_backtest_summary(results)

    # 資金推移をプロット
    plot_path = os.path.join(model_dir, "bankroll_history.png")
    plot_bankroll_history(results, save_path=plot_path)
    print(f"\nBankroll history plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
