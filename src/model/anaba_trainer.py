"""穴馬予測ヘッド (LightGBM) - 学習と推論。

穴馬: rank == 1 AND pop >= min_pop (デフォルト 5)
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


def get_anaba_feature_columns(
    base_columns: list[str],
    ts_columns: list[str],
    use_ts_odds: bool = True,
    extra_base_columns: list[str] | None = None,
) -> list[str]:
    """穴馬モデル用の特徴量列を組み立てる (順序維持・重複除去)。"""
    cols = list(base_columns)
    if extra_base_columns:
        cols = cols + list(extra_base_columns)
    if use_ts_odds:
        cols = cols + list(ts_columns) + ["has_ts_odds"]
    seen: set[str] = set()
    result: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def save_anaba_model(model: lgb.Booster, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def load_anaba_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def save_anaba_meta(meta: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def load_anaba_meta(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# --- 学習関数 (将来の再学習で利用; 推論のみなら不要) ---

def _cat_indices(feature_columns: list[str]) -> list[int]:
    return [i for i, c in enumerate(feature_columns) if c in CATEGORICAL_FEATURES]


def train_anaba_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    cfg: dict,
    feature_columns: list[str],
):
    """穴馬ヘッドを LightGBM + Optuna で学習。"""
    import optuna
    from sklearn.metrics import average_precision_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    cat_indices = _cat_indices(feature_columns)
    search = cfg["model"]["search_space"]
    max_iter = cfg["model"]["max_iterations"]
    early_stop = cfg["model"]["early_stopping_rounds"]
    seed = cfg["model"]["seed"]

    train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=cat_indices, free_raw_data=False)
    valid_data = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cat_indices, reference=train_data, free_raw_data=False)

    def objective(trial):
        boosting_type = trial.suggest_categorical("boosting_type", search["boosting_type"])
        subsample_val = trial.suggest_float("subsample", *search["subsample"])
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "verbosity": -1, "feature_pre_filter": False, "seed": seed,
            "boosting_type": boosting_type,
            "learning_rate": trial.suggest_float("learning_rate", *search["learning_rate"], log=True),
            "num_leaves": trial.suggest_int("num_leaves", *search["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *search["max_depth"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *search["min_child_samples"]),
            "subsample": subsample_val,
            "subsample_freq": 1 if subsample_val < 1.0 else 0,
            "colsample_bytree": trial.suggest_float("colsample_bytree", *search["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *search["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *search["reg_lambda"], log=True),
            "scale_pos_weight": spw,
        }
        callbacks = [lgb.log_evaluation(period=0)]
        if boosting_type != "dart":
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, verbose=False))
        model = lgb.train(params, train_data, num_boost_round=max_iter, valid_sets=[valid_data], callbacks=callbacks)
        best_iter = model.best_iteration if model.best_iteration and model.best_iteration > 0 else model.current_iteration()
        trial.set_user_attr("best_iteration", best_iter)
        try:
            ap = average_precision_score(valid_y, model.predict(valid_x))
        except ValueError:
            ap = 0.0
        return -ap

    anaba_cfg = cfg.get("anaba", {})
    n_trials = anaba_cfg.get("optuna_n_trials", cfg["model"]["optuna"]["n_trials"])
    timeout = anaba_cfg.get("optuna_timeout", cfg["model"]["optuna"].get("timeout"))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"  Best -AP: {study.best_value:.6f} (AP={-study.best_value:.4f})")

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary", "metric": "binary_logloss",
        "verbosity": -1, "seed": cfg["model"]["seed"],
        "scale_pos_weight": spw,
    })
    full_x = pd.concat([train_x, valid_x], axis=0)
    full_y = pd.concat([train_y, valid_y], axis=0)
    train_data = lgb.Dataset(full_x, label=full_y, categorical_feature=cat_indices)
    best_model = lgb.train(
        best_params, train_data,
        num_boost_round=study.best_trial.user_attrs.get("best_iteration", cfg["model"]["max_iterations"]),
    )
    return best_model, study


def evaluate_anaba(model, x, y) -> dict:
    """AP / AUC / Hit@K で評価。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(y) == 0:
        return {"ap": 0.0, "auc": 0.0, "positive_rate": 0.0, "n": 0}
    probs = model.predict(x)
    pos_rate = float(y.mean())
    try:
        ap = float(average_precision_score(y, probs))
    except ValueError:
        ap = 0.0
    try:
        auc = float(roc_auc_score(y, probs)) if 0 < pos_rate < 1 else 0.0
    except ValueError:
        auc = 0.0
    n_pos = int(y.sum())
    hit_at_k = 0.0
    if n_pos > 0:
        order = np.argsort(probs)[::-1][:n_pos]
        hit_at_k = float(y.iloc[order].sum() / n_pos)
    return {"ap": ap, "auc": auc, "positive_rate": pos_rate, "n": int(len(y)), "hit_at_k_positives": hit_at_k}
