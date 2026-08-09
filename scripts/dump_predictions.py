#!/usr/bin/env python3
"""テスト期間全体の予測値をダンプする (戦略チューニング用)。

特徴量エンジニアリング (~10分) を一度だけ実行し、
race_id / 着順 / 確定オッズ / 校正後予測確率 を CSV に保存する。
以降の戦略パラメータのグリッドサーチはこの CSV 上で瞬時に回せる
(tune_strategy.py が消費する。「予測とデシジョンの分離」パターン)。

【レビュー時のチェックポイント】
出力 CSV の期間が「2024-07 以降のみ」であること。
2023 年以前の行が混ざっていたら pipeline.transform の行順保存が
壊れている兆候 (過去に実際この方法でバグを検出した)。

使い方:
    python scripts/dump_predictions.py --config config/backtest_masters_prod.yaml \
        --out models_masters/test_predictions.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config
from src.features.pipeline import FeaturePipeline, FEATURE_COLUMNS, assert_no_market_info
from src.features.market import add_market_features
from src.model.calibrator import HoldoutCalibrator
from src.model.win_trainer import load_win_meta, load_win_model
from src.model.trainer import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/backtest_masters_prod.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg["model"]["dir"]
    out_path = args.out or os.path.join(model_dir, "test_predictions.csv")

    print("Loading models...")
    win_model = load_win_model(os.path.join(model_dir, "win_model.txt"))
    win_meta = load_win_meta(os.path.join(model_dir, "win_meta.pkl"))
    show_model = load_model(os.path.join(model_dir, "lgbm_model.txt"))
    show_cal = HoldoutCalibrator.load(os.path.join(model_dir, "calibrator.pkl"))

    print("Loading data + feature engineering (this takes a while)...")
    train_df, valid_df, test_df = load_all_data(cfg)
    pipeline = FeaturePipeline.load(os.path.join(model_dir, "pipeline.pkl"), cfg)
    all_data = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    transformed = pipeline.transform(all_data)
    n_train, n_valid = len(train_df), len(valid_df)
    test_feat = transformed.iloc[n_train + n_valid:].reset_index(drop=True)
    print(f"Test rows: {len(test_feat)}")

    for c in win_meta["feature_columns"]:
        if c not in test_feat.columns:
            test_feat[c] = 0.0
    assert_no_market_info(win_meta["feature_columns"])

    print("Predicting...")
    raw_win = win_model.predict(test_feat[win_meta["feature_columns"]])
    test_feat["win_pred_prob_raw"] = raw_win

    win_cal_path = os.path.join(model_dir, "win_calibrator.pkl")
    if os.path.exists(win_cal_path):
        win_cal = HoldoutCalibrator.load(win_cal_path)
        test_feat["win_pred_prob"] = win_cal.predict(raw_win)
    else:
        test_feat["win_pred_prob"] = raw_win

    avail = [c for c in FEATURE_COLUMNS if c in test_feat.columns]
    test_feat["show_pred_prob"] = show_cal.predict(show_model.predict(test_feat[avail]))

    # 市場残差ヘッド (存在すれば)。市場特徴量を計算してから予測・校正する。
    market_model_path = os.path.join(model_dir, "win_market_model.txt")
    market_meta_path = os.path.join(model_dir, "win_market_meta.pkl")
    if os.path.exists(market_model_path) and os.path.exists(market_meta_path):
        print("Predicting with MARKET-RESIDUAL head...")
        m_model = load_win_model(market_model_path)
        m_meta = load_win_meta(market_meta_path)
        test_m = add_market_features(test_feat)
        for c in m_meta["feature_columns"]:
            if c not in test_m.columns:
                test_m[c] = 0.0
        raw_m = m_model.predict(test_m[m_meta["feature_columns"]])
        test_feat["market_prob"] = test_m["market_prob"]
        m_cal_path = os.path.join(model_dir, "win_market_calibrator.pkl")
        if os.path.exists(m_cal_path):
            m_cal = HoldoutCalibrator.load(m_cal_path)
            test_feat["win_market_pred_prob"] = m_cal.predict(raw_m)
        else:
            test_feat["win_market_pred_prob"] = raw_m
    else:
        print("(market head not found — skipping)")

    cols = [
        "race_id", "year", "month", "day", "place", "race_num", "horse_num",
        "horse", "rank", "error_code", "pop", "win_odds",
        "win_pred_prob_raw", "win_pred_prob", "show_pred_prob",
        "market_prob", "win_market_pred_prob",
    ]
    cols = [c for c in cols if c in test_feat.columns]
    test_feat[cols].to_csv(out_path, index=False)
    print(f"Saved {len(test_feat)} rows to {out_path}")


if __name__ == "__main__":
    main()
