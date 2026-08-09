"""単勝予測ヘッド (LightGBM) - 学習と推論。

================================================================================
【レビューガイド】設計判断の理由
================================================================================
AI 競馬予想マスターズ 2026 の主戦場である「単勝 1 着」を予測する。

■ なぜ複勝モデル (trainer.py) と別ヘッドにするか
  - 複勝 (rank<=3, 陽性率 21%) と単勝 (rank==1, 陽性率 7%) はターゲットの
    性質が違う。「3 着に入る堅実さ」と「勝ち切る力」は別のシグナル。
  - 1 つのモデルで両方をやると多数派 (複勝的な安定馬) に引っ張られ、
    単勝的な「勝ち切る」シグナルが埋もれる。

■ クラス不均衡対応: scale_pos_weight = neg/pos (≒ 13)
  - 陽性 (勝ち馬) の勾配を 13 倍にして、少数派を軽視しないようにする。
  - 副作用: 出力確率が実際の的中率より高めに歪む (posterior shift)。
    → 学習後に Isotonic Regression で校正する (scripts/train.py 側)。
    校正前の生確率は 0.07-0.14 に圧縮されており、EV 計算には使えない点に注意。

■ Optuna の目的関数: Brier Score (= 確率の二乗誤差) を最小化
  - AUC ではなく Brier を選ぶ理由: このモデルの出力は「賭け金計算 (EV = p×odds)」
    に直接使うため、順位の正しさ (AUC) より確率値の正確さが重要。
  - LogLoss でなく Brier: LogLoss は極端な確率の外れに過大なペナルティを与え、
    不均衡データではノイズに敏感になりがち。Brier の方が安定。

■ 最終モデルは train+valid を結合して再学習
  - Optuna でベストパラメータと best_iteration を決めた後、valid も学習に
    使い切る (valid はハイパラ選択に使い終わったので、捨てるのはもったいない)。
  - best_iteration を固定して再学習するので early stopping 用の valid は不要。
================================================================================
"""

from __future__ import annotations

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from src.features.pipeline import CATEGORICAL_FEATURES
except Exception:
    CATEGORICAL_FEATURES: list[str] = []


def _cat_indices(feature_columns: list[str]) -> list[int]:
    return [i for i, c in enumerate(feature_columns) if c in CATEGORICAL_FEATURES]


def train_win_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    cfg: dict,
    feature_columns: list[str],
):
    """単勝ヘッドを LightGBM + Optuna で学習する。

    - 目的関数: Brier Score 最小化
    - クラス不均衡対応: scale_pos_weight (neg/pos)
    - 早期打ち切り: valid Brier が改善しなくなったら終了
    """
    import optuna
    from sklearn.metrics import brier_score_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # scale_pos_weight = 陰性数/陽性数 (陽性率 7% → 約 13)。
    # 全 trial で共通の固定値とし、探索空間には含めない
    # (探索すると Brier 最適化と確率スケールの歪みが交絡して不安定になるため)。
    #
    # ★ use_scale_pos_weight=false (市場残差ヘッド用):
    #   市場確率を特徴量に持つモデルでは spw が逆効果になりうる。
    #   spw で確率スケールを 13 倍歪ませてから Isotonic で戻す過程で
    #   微小な残差エッジが失われるため、spw=1 (歪ませない) の選択肢を用意する。
    #   Brier 最適化は不均衡でも適切に機能するので spw なしでも学習は成立する。
    win_cfg_local = cfg.get("win", {})
    use_spw = bool(win_cfg_local.get("use_scale_pos_weight", True))
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    spw = (neg / pos) if (pos > 0 and use_spw) else 1.0
    print(f"  scale_pos_weight: {spw:.2f} ({'有効' if use_spw else '無効化 (市場残差用)'})")

    cat_indices = _cat_indices(feature_columns)
    search = cfg["model"]["search_space"]
    max_iter = cfg["model"]["max_iterations"]
    early_stop = cfg["model"]["early_stopping_rounds"]
    seed = cfg["model"]["seed"]

    # Dataset は trial 間で再利用する (free_raw_data=False)。
    # 毎 trial で作り直すと 1.4M 行のビニングが繰り返されて遅い。
    train_data = lgb.Dataset(
        train_x, label=train_y, categorical_feature=cat_indices, free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        valid_x, label=valid_y, categorical_feature=cat_indices,
        reference=train_data, free_raw_data=False,
    )

    def objective(trial):
        boosting_type = trial.suggest_categorical("boosting_type", search["boosting_type"])
        subsample_val = trial.suggest_float("subsample", *search["subsample"])
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "feature_pre_filter": False,
            "seed": seed,
            "boosting_type": boosting_type,
            "learning_rate": trial.suggest_float(
                "learning_rate", *search["learning_rate"], log=True,
            ),
            "num_leaves": trial.suggest_int("num_leaves", *search["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *search["max_depth"]),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *search["min_child_samples"],
            ),
            "subsample": subsample_val,
            "subsample_freq": 1 if subsample_val < 1.0 else 0,
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search["colsample_bytree"],
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", *search["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *search["reg_lambda"], log=True),
            "scale_pos_weight": spw,
        }
        # early stopping: valid の logloss が early_stop 回連続で改善しなければ打ち切り。
        # dart は木の drop により評価が単調でなく early stopping と相性が悪いので除外。
        callbacks = [lgb.log_evaluation(period=0)]
        if boosting_type != "dart":
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, verbose=False))
        model = lgb.train(
            params, train_data, num_boost_round=max_iter,
            valid_sets=[valid_data], callbacks=callbacks,
        )
        # best_iteration を trial に記録 → 最終再学習で同じ本数だけ木を作る
        best_iter = (
            model.best_iteration if model.best_iteration and model.best_iteration > 0
            else model.current_iteration()
        )
        trial.set_user_attr("best_iteration", best_iter)
        # Optuna が最小化するのは valid Brier (学習中の metric=logloss とは別物。
        # logloss は early stopping 用、Brier はハイパラ選択用と役割分担)
        preds = model.predict(valid_x)
        return brier_score_loss(valid_y, preds)

    win_cfg = cfg.get("win", {})
    n_trials = win_cfg.get("optuna_n_trials", cfg["model"]["optuna"]["n_trials"])
    timeout = win_cfg.get("optuna_timeout", cfg["model"]["optuna"].get("timeout"))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"  Best Brier: {study.best_value:.6f}")

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": cfg["model"]["seed"],
        "scale_pos_weight": spw,
    })
    full_x = pd.concat([train_x, valid_x], axis=0)
    full_y = pd.concat([train_y, valid_y], axis=0)
    train_data = lgb.Dataset(full_x, label=full_y, categorical_feature=cat_indices)
    best_model = lgb.train(
        best_params, train_data,
        num_boost_round=study.best_trial.user_attrs.get(
            "best_iteration", cfg["model"]["max_iterations"],
        ),
    )
    return best_model, study


def evaluate_win(model, x, y) -> dict:
    """単勝ヘッドを Brier / AUC / Top-K hit-rate で評価する。"""
    from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

    if len(y) == 0:
        return {"brier": 0.0, "auc": 0.0, "ap": 0.0, "positive_rate": 0.0, "n": 0}
    probs = model.predict(x)
    pos_rate = float(y.mean())
    metrics = {
        "brier": float(brier_score_loss(y, probs)),
        "positive_rate": pos_rate,
        "n": int(len(y)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y, probs)) if 0 < pos_rate < 1 else 0.0
    except ValueError:
        metrics["auc"] = 0.0
    try:
        metrics["ap"] = float(average_precision_score(y, probs))
    except ValueError:
        metrics["ap"] = 0.0

    # Top-K hit@K (K = 陽性数)
    n_pos = int(y.sum())
    if n_pos > 0:
        order = np.argsort(probs)[::-1][:n_pos]
        metrics["hit_at_k_positives"] = float(y.iloc[order].sum() / n_pos)
    else:
        metrics["hit_at_k_positives"] = 0.0
    return metrics


def save_win_model(model: lgb.Booster, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def load_win_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def save_win_meta(meta: dict, path: str) -> None:
    """単勝モデルのメタ情報 (feature_columns, test_metrics 等) を保存する。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def load_win_meta(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
