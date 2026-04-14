"""モデルの評価指標と可視化。"""

from __future__ import annotations

import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.model.calibrator import HoldoutCalibrator


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """AUC、Brierスコア、対数損失を計算する。"""
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """各閾値における的中率と(任意で)回収率を計算する。"""
    if thresholds is None:
        thresholds = np.arange(0.25, 0.80, 0.05)

    rows = []
    for t in thresholds:
        mask = y_prob >= t
        if mask.sum() < 10:
            continue
        hits = y_true[mask].sum()
        total = mask.sum()
        hit_rate = hits / total

        row = {"threshold": round(t, 3), "n_bets": total,
               "hits": int(hits), "hit_rate": round(hit_rate, 4)}

        if odds is not None:
            # シンプルな回収率: sum(odds * hit) / n_bets
            payoff = (odds[mask] * y_true[mask]).sum()
            row["recovery_rate"] = round(payoff / total, 4)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
    save_path: str | None = None,
) -> None:
    """生の予測とキャリブレーション後の予測を比較するキャリブレーション曲線を描画する。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for label, probs in [("Raw", y_prob_raw), ("Calibrated", y_prob_calibrated)]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model: lgb.Booster,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """LightGBMの特徴量重要度(gain)を描画する。"""
    importance = model.feature_importance(importance_type="gain")
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in indices],
        importance[indices],
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model: lgb.Booster,
    calibrator: HoldoutCalibrator,
    eval_x: pd.DataFrame,
    eval_y: pd.Series,
    feature_names: list[str],
    output_dir: str = "./models",
    odds: np.ndarray | None = None,
) -> dict:
    """評価一式を実行し、プロットを保存する。

    Args:
        odds: eval_x/eval_yに対応する払戻オッズの配列。
              指定された場合、閾値分析に回収率が含まれる。

    評価指標の辞書を返す。
    """
    raw_probs = model.predict(eval_x)
    cal_probs = calibrator.predict(raw_probs)

    metrics_raw = compute_metrics(eval_y.values, raw_probs)
    metrics_cal = compute_metrics(eval_y.values, cal_probs)

    print("=" * 50)
    print("Raw model metrics:")
    for k, v in metrics_raw.items():
        print(f"  {k}: {v:.4f}")
    print("Calibrated model metrics:")
    for k, v in metrics_cal.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    # プロット
    plot_calibration_curve(
        eval_y.values, raw_probs, cal_probs,
        save_path=os.path.join(output_dir, "calibration_curve.png"),
    )
    plot_feature_importance(
        model, feature_names,
        save_path=os.path.join(output_dir, "feature_importance.png"),
    )

    # 閾値分析
    thr_df = threshold_analysis(eval_y.values, cal_probs, odds=odds)
    print("\nThreshold analysis:")
    print(thr_df.to_string(index=False))

    return {
        "raw": metrics_raw,
        "calibrated": metrics_cal,
        "threshold_analysis": thr_df,
    }
