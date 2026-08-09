#!/usr/bin/env python3
"""学習パイプラインのエントリーポイント。

使い方:
    python scripts/train.py --config config/masters_2026.yaml

================================================================================
【レビューガイド】学習フロー全体像 (Step 1-7)
================================================================================
Step 1  データ読み込み: train (1995-2023) / valid (2024前半) / test (2024後半+2025)
        の時系列 3 分割。ランダム分割は使わない (同一馬・同一開催の情報が
        分割をまたいで漏れるため、時間で切るのが鉄則)。
Step 2  特徴量: pipeline.fit(train_df) → 統計量は train のみから算出。
        transform は train+valid+test を縦結合して一度に実行
        (ローリング特徴量を年またぎで正しく計算するため)。
        変換後に iloc で元の 3 分割へ戻す — transform の
        「行順保存の不変条件」に依存している (pipeline.py 参照)。
Step 3  複勝モデル: Optuna でハイパラ探索 (valid Brier 最小化) → 最終再学習。
Step 4  複勝キャリブレーション: valid の前半 50% で Isotonic を fit。
        「学習に使ったデータで校正すると過信がそのまま残る」ため、
        必ずモデルが見ていないデータで校正する。
Step 5  評価: 校正済み確率で valid 後半 + test を評価 (AUC/Brier/LogLoss)。
Step 6  複勝側の成果物保存 (lgbm_model.txt / calibrator.pkl / pipeline.pkl)。
Step 7  単勝ヘッド: 同じ特徴量行列でターゲットだけ rank==1 に替えて別モデルを学習。
        + Isotonic 校正 (win_calibrator.pkl)。校正の効果は劇的で、
        scale_pos_weight による圧縮確率 (0.07-0.14) が実的中率スケール
        (0.00-0.36) に展開され、EV = p×odds の計算が意味を持つようになる。
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_all_data, load_config, filter_errors
from src.features.pipeline import (
    FeaturePipeline,
    FEATURE_COLUMNS,
    assert_no_market_info,
    build_target,
    build_target_win,
)
from src.features.market import MARKET_FEATURE_COLUMNS, add_market_features
from src.model.trainer import train_model, save_model
from src.model.calibrator import calibrate_model
from src.model.evaluator import evaluate_model
from src.model.win_trainer import (
    evaluate_win,
    save_win_meta,
    save_win_model,
    train_win_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train keiba prediction model")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--data-dir", default=None, help="Override data directory from config")
    parser.add_argument("--market-only", action="store_true",
                        help="市場残差ヘッド (Step 8) のみ学習する。"
                             "既存のオッズなしヘッド (Step 3-7) はスキップ。"
                             "特徴量エンジニアリングは必要なので所要 ~30-40 分")
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
    # (例: 2024 年の馬の「直近 3 走」には 2023 年のレースが含まれるため、
    #  年ごとに別々に transform すると履歴が切れてしまう)
    all_data = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    pipeline.fit(train_df)
    all_transformed = pipeline.transform(all_data)

    # 分割し直す。concat 順 = [train | valid | test] であり、
    # transform が行順を保存する (pipeline.py の不変条件) ため iloc で戻せる。
    n_train = len(train_df)
    n_valid = len(valid_df)
    train_feat = all_transformed.iloc[:n_train].reset_index(drop=True)
    valid_feat = all_transformed.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_feat = all_transformed.iloc[n_train + n_valid:].reset_index(drop=True)

    # 特徴量行列を取得
    available_features = [c for c in FEATURE_COLUMNS if c in train_feat.columns]

    # ★ AI 競馬予想マスターズ: オッズ/人気 が混入していないことをアサート
    assert_no_market_info(available_features)

    print(f"  Features: {len(available_features)} columns")
    print(f"  Feature names: {available_features}")

    train_x = train_feat[available_features]
    train_y = build_target(train_feat)
    valid_x = valid_feat[available_features]
    valid_y = build_target(valid_feat)
    test_x = test_feat[available_features]
    test_y = build_target(test_feat)

    print(f"  Train positive rate (複勝): {train_y.mean():.3f}")
    print(f"  Valid positive rate (複勝): {valid_y.mean():.3f}")

    # --market-only: オッズなしヘッド (Step 3-7) を飛ばして市場残差ヘッドだけ学習
    if args.market_only:
        print("\n[--market-only] Steps 3-7 をスキップし、市場残差ヘッドのみ学習します")
        train_market_head(
            cfg=cfg,
            train_feat=train_feat,
            valid_feat=valid_feat,
            test_feat=test_feat,
            model_dir=model_dir,
        )
        print("\nTraining complete!")
        return

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

    # ================================================================
    # ステップ 7+: 単勝予測ヘッド (AI 競馬予想マスターズ 2026 の主戦場)
    # ================================================================
    win_cfg = cfg.get("win", {})
    if win_cfg.get("enabled", True):  # デフォルトで有効化
        train_win_head(
            cfg=cfg,
            train_feat=train_feat,
            valid_feat=valid_feat,
            test_feat=test_feat,
            available_features=available_features,
            model_dir=model_dir,
        )

    # ================================================================
    # ステップ 8: 市場残差ヘッド (実マネー運用向けの本命モデル)
    # ================================================================
    market_cfg = cfg.get("market", {})
    if market_cfg.get("enabled", False):
        train_market_head(
            cfg=cfg,
            train_feat=train_feat,
            valid_feat=valid_feat,
            test_feat=test_feat,
            model_dir=model_dir,
        )

    print("\nTraining complete!")


def train_market_head(
    cfg: dict,
    train_feat: pd.DataFrame,
    valid_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    model_dir: str,
) -> None:
    """市場残差ヘッド (単勝, オッズ特徴量あり) を学習・校正・保存する。

    - ターゲット: rank == 1 (win ヘッドと同じ)
    - 特徴量: FEATURE_COLUMNS (46) + MARKET_FEATURE_COLUMNS (3) = 49 列
    - 狙い: 市場確率をベースラインに、基礎能力特徴量で残差を修正する。
      「校正後確率 × オッズ ≥ 閾値」の馬だけに賭ける実マネー戦略の中核。
    - 成否の判定基準: モデルの Brier が market_prob 単体の Brier を
      下回るか (= 市場に対して情報を足せているか) を必ず確認する。
    """
    from sklearn.metrics import brier_score_loss

    print("\n" + "=" * 60)
    print("Step 8: Training MARKET-RESIDUAL head (win + odds features)...")
    print("=" * 60)

    # 市場特徴量を付与 (win_odds は transform 後も原列として残っている)
    train_m = add_market_features(train_feat)
    valid_m = add_market_features(valid_feat)
    test_m = add_market_features(test_feat)

    feature_columns = [c for c in FEATURE_COLUMNS if c in train_m.columns] + MARKET_FEATURE_COLUMNS
    print(f"  Market head features: {len(feature_columns)} columns "
          f"(base {len(FEATURE_COLUMNS)} + market {len(MARKET_FEATURE_COLUMNS)})")

    train_x = train_m[feature_columns]
    valid_x = valid_m[feature_columns]
    test_x = test_m[feature_columns]
    train_y = build_target_win(train_m)
    valid_y = build_target_win(valid_m)
    test_y = build_target_win(test_m)

    # ベースライン: 市場確率そのものの Brier (モデルはこれを下回らないと存在価値がない)
    market_brier_valid = float(brier_score_loss(valid_y, valid_m["market_prob"].clip(0, 1)))
    market_brier_test = float(brier_score_loss(test_y, test_m["market_prob"].clip(0, 1)))
    print(f"  [baseline] market_prob Brier: valid={market_brier_valid:.5f} test={market_brier_test:.5f}")

    # win ヘッドと同じ学習ルーチンを流用 (Optuna trials 数は market: セクションで指定可)
    market_train_cfg = dict(cfg)
    market_train_cfg["win"] = cfg.get("market", {})  # optuna_n_trials / timeout を引き回す
    market_model, market_study = train_win_model(
        train_x, train_y, valid_x, valid_y, market_train_cfg, feature_columns,
    )

    print("\n  Calibrating market head (Isotonic on valid holdout)...")
    market_calibrator, _, _ = calibrate_model(
        market_model, valid_x, valid_y,
        holdout_fraction=cfg["calibration"]["holdout_fraction"],
        seed=cfg["model"]["seed"],
    )

    print("\n  Market head test set evaluation (raw):")
    test_metrics = evaluate_win(market_model, test_x, test_y)
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    cal_test = market_calibrator.predict(market_model.predict(test_x))
    cal_brier = float(brier_score_loss(test_y, cal_test))
    print(f"    brier (calibrated): {cal_brier:.5f}")
    print(f"    vs market baseline: {market_brier_test:.5f} "
          f"({'✓ モデルが市場を上回る' if cal_brier < market_brier_test else '✗ 市場に勝てていない'})")

    save_win_model(market_model, os.path.join(model_dir, "win_market_model.txt"))
    market_calibrator.save(os.path.join(model_dir, "win_market_calibrator.pkl"))
    save_win_meta(
        {
            "feature_columns": feature_columns,
            "target": "rank == 1",
            "test_metrics": test_metrics,
            "test_brier_calibrated": cal_brier,
            "test_brier_market_baseline": market_brier_test,
            "best_params": market_study.best_params,
        },
        os.path.join(model_dir, "win_market_meta.pkl"),
    )
    print(f"  Market model saved to: {model_dir}/win_market_model.txt")
    print(f"  Market calibrator saved to: {model_dir}/win_market_calibrator.pkl")


def train_win_head(
    cfg: dict,
    train_feat: pd.DataFrame,
    valid_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    available_features: list[str],
    model_dir: str,
) -> None:
    """単勝予測ヘッドを学習し、win_model.txt / win_meta.pkl に保存する。

    - ターゲット: rank == 1 (陽性率 ~7%)
    - 特徴量: 複勝モデルと同じ FEATURE_COLUMNS (オッズ/人気 なし)
    - クラス不均衡対応: scale_pos_weight
    - 目的関数: Brier Score
    """
    print("\n" + "=" * 60)
    print("Step 7: Training WIN (単勝) head for AI 競馬予想マスターズ...")
    print("=" * 60)

    # 特徴量アサート (念のためもう一度)
    assert_no_market_info(available_features)

    train_x = train_feat[available_features]
    valid_x = valid_feat[available_features]
    test_x = test_feat[available_features]
    train_y = build_target_win(train_feat)
    valid_y = build_target_win(valid_feat)
    test_y = build_target_win(test_feat)

    print(f"  Train positive rate (単勝): {train_y.mean():.3f}")
    print(f"  Valid positive rate (単勝): {valid_y.mean():.3f}")

    if int(train_y.sum()) == 0:
        print("  ERROR: No positive win samples. Skip win head training.")
        return

    win_model, win_study = train_win_model(
        train_x, train_y, valid_x, valid_y, cfg, available_features,
    )

    # 単勝ヘッドのキャリブレーション (Isotonic Regression)
    # 生確率が scale_pos_weight の影響で 0.07-0.14 に圧縮されるため、
    # 実的中率スケールに補正して EV = prob × odds の計算精度を上げる。
    print("\n  Calibrating win head (Isotonic on valid holdout)...")
    win_calibrator, _, _ = calibrate_model(
        win_model, valid_x, valid_y,
        holdout_fraction=cfg["calibration"]["holdout_fraction"],
        seed=cfg["model"]["seed"],
    )

    print("\n  Win head test set evaluation (raw):")
    test_metrics = evaluate_win(win_model, test_x, test_y)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # キャリブレーション後の Brier も確認
    from sklearn.metrics import brier_score_loss
    cal_test_probs = win_calibrator.predict(win_model.predict(test_x))
    cal_brier = float(brier_score_loss(test_y, cal_test_probs))
    print(f"    brier (calibrated): {cal_brier:.4f}")
    print(f"    calibrated prob range: [{cal_test_probs.min():.4f}, {cal_test_probs.max():.4f}]")

    save_win_model(win_model, os.path.join(model_dir, "win_model.txt"))
    win_calibrator.save(os.path.join(model_dir, "win_calibrator.pkl"))
    save_win_meta(
        {
            "feature_columns": available_features,
            "target": "rank == 1",
            "test_metrics": test_metrics,
            "test_brier_calibrated": cal_brier,
            "best_params": win_study.best_params,
        },
        os.path.join(model_dir, "win_meta.pkl"),
    )
    print(f"  Win model saved to: {model_dir}/win_model.txt")
    print(f"  Win calibrator saved to: {model_dir}/win_calibrator.pkl")
    print(f"  Win meta saved to:  {model_dir}/win_meta.pkl")


if __name__ == "__main__":
    main()
