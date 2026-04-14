#!/usr/bin/env python3
"""学習パイプラインのエントリーポイント。

使い方:
    python scripts/train.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config, filter_errors
from src.features.pipeline import FeaturePipeline, FEATURE_COLUMNS, build_target
from src.model.trainer import train_model, save_model
from src.model.calibrator import calibrate_model
from src.model.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train keiba prediction model")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--data-dir", default=None, help="Override data directory from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    model_dir = cfg["model"]["dir"]
    os.makedirs(model_dir, exist_ok=True)

    # ステップ1: データ読み込み
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)
    train_df, valid_df, test_df = load_all_data(cfg)
    print(f"  Train: {len(train_df)} rows")
    print(f"  Valid: {len(valid_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    # 学習データからエラー行を除外
    train_df = filter_errors(train_df)
    print(f"  Train after error filter: {len(train_df)} rows")

    # ステップ2: 特徴量エンジニアリング
    print("\n" + "=" * 60)
    print("Step 2: Feature engineering...")
    print("=" * 60)

    pipeline = FeaturePipeline(cfg)

    # 学習データでfit（騎手・種牡馬の統計量を計算）
    # その後、全データをまとめてtransformしてローリング特徴量を正しく算出
    all_data = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    pipeline.fit(train_df)
    all_transformed = pipeline.transform(all_data)

    # 分割し直す
    n_train = len(train_df)
    n_valid = len(valid_df)
    train_feat = all_transformed.iloc[:n_train].reset_index(drop=True)
    valid_feat = all_transformed.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_feat = all_transformed.iloc[n_train + n_valid:].reset_index(drop=True)

    # 特徴量行列を取得
    available_features = [c for c in FEATURE_COLUMNS if c in train_feat.columns]
    print(f"  Features: {len(available_features)} columns")
    print(f"  Feature names: {available_features}")

    train_x = train_feat[available_features]
    train_y = build_target(train_feat)
    valid_x = valid_feat[available_features]
    valid_y = build_target(valid_feat)
    test_x = test_feat[available_features]
    test_y = build_target(test_feat)

    print(f"  Train positive rate: {train_y.mean():.3f}")
    print(f"  Valid positive rate: {valid_y.mean():.3f}")

    # ステップ3: Optunaを用いたモデル学習
    print("\n" + "=" * 60)
    print("Step 3: Training LightGBM with Optuna...")
    print("=" * 60)

    model, study = train_model(
        train_x, train_y, valid_x, valid_y, cfg, available_features
    )

    # ステップ4: ホールドアウトによるキャリブレーション
    print("\n" + "=" * 60)
    print("Step 4: Calibrating on holdout split...")
    print("=" * 60)

    calibrator, eval_x, eval_y = calibrate_model(
        model, valid_x, valid_y,
        holdout_fraction=cfg["calibration"]["holdout_fraction"],
        seed=cfg["model"]["seed"],
    )

    # ステップ5: 評価
    print("\n" + "=" * 60)
    print("Step 5: Evaluation...")
    print("=" * 60)

    metrics = evaluate_model(
        model, calibrator, eval_x, eval_y, available_features, model_dir
    )

    # テストセットに対しても評価
    print("\nTest set evaluation:")
    raw_test = model.predict(test_x)
    cal_test = calibrator.predict(raw_test)
    from src.model.evaluator import compute_metrics
    test_metrics = compute_metrics(test_y.values, cal_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ステップ6: モデル・パイプラインを保存
    print("\n" + "=" * 60)
    print("Step 6: Saving model and pipeline...")
    print("=" * 60)

    save_model(model, os.path.join(model_dir, "lgbm_model.txt"))
    calibrator.save(os.path.join(model_dir, "calibrator.pkl"))
    pipeline.save(os.path.join(model_dir, "pipeline.pkl"))

    # 参考として騎手・種牡馬の統計量をCSVで保存
    pipeline.jockey_stats.to_csv(os.path.join(model_dir, "jockey_stats.csv"), index=False)
    pipeline.sire_stats.to_csv(os.path.join(model_dir, "sire_stats.csv"), index=False)

    print(f"  Model saved to: {model_dir}/lgbm_model.txt")
    print(f"  Calibrator saved to: {model_dir}/calibrator.pkl")
    print(f"  Pipeline saved to: {model_dir}/pipeline.pkl")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
