"""Optunaによるハイパーパラメータ最適化を用いたLightGBMモデルの学習。"""

from __future__ import annotations

import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.features.pipeline import CATEGORICAL_FEATURES


def _get_categorical_indices(feature_columns: list[str]) -> list[int]:
    """特徴量列リスト内のカテゴリ特徴量のインデックスを取得する。"""
    return [i for i, c in enumerate(feature_columns) if c in CATEGORICAL_FEATURES]


def create_objective(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    cfg: dict,
    feature_columns: list[str],
) -> callable:
    """Optunaの目的関数を生成する。

    検証セット上のBrierスコアを最適化する。
    """
    cat_indices = _get_categorical_indices(feature_columns)
    search = cfg["model"]["search_space"]
    max_iter = cfg["model"]["max_iterations"]
    early_stop = cfg["model"]["early_stopping_rounds"]
    seed = cfg["model"]["seed"]

    train_data = lgb.Dataset(
        train_x, label=train_y,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        valid_x, label=valid_y,
        categorical_feature=cat_indices,
        reference=train_data,
        free_raw_data=False,
    )

    def objective(trial: optuna.Trial) -> float:
        boosting_type = trial.suggest_categorical(
            "boosting_type", search["boosting_type"]
        )
        subsample_val = trial.suggest_float("subsample", *search["subsample"])

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "feature_pre_filter": False,
            "seed": seed,
            "boosting_type": boosting_type,
            "learning_rate": trial.suggest_float(
                "learning_rate", *search["learning_rate"], log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", *search["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *search["max_depth"]),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *search["min_child_samples"]
            ),
            "subsample": subsample_val,
            "subsample_freq": 1 if subsample_val < 1.0 else 0,
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search["colsample_bytree"]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *search["reg_alpha"], log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search["reg_lambda"], log=True
            ),
        }

        # DARTはearly stoppingを安定的にサポートしない
        use_early_stop = boosting_type != "dart"

        callbacks = [lgb.log_evaluation(period=0)]
        if use_early_stop:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False)
            )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_iter,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration if model.best_iteration > 0 else model.current_iteration()
        trial.set_user_attr("best_iteration", best_iter)

        preds = model.predict(valid_x)
        brier = brier_score_loss(valid_y, preds)
        return brier

    return objective


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    cfg: dict,
    feature_columns: list[str],
) -> tuple[lgb.Booster, optuna.Study]:
    """Optuna最適化を実行し、最良パラメータで最終モデルを学習する。

    Returns:
        (best_model, study)
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = create_objective(
        train_x, train_y, valid_x, valid_y, cfg, feature_columns
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=cfg["model"]["optuna"]["n_trials"],
        timeout=cfg["model"]["optuna"].get("timeout"),
    )

    print(f"Best Brier score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # 最良パラメータでtrain+validの全データを使って再学習する
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": cfg["model"]["seed"],
    })

    cat_indices = _get_categorical_indices(feature_columns)
    full_x = pd.concat([train_x, valid_x], axis=0)
    full_y = pd.concat([train_y, valid_y], axis=0)

    train_data = lgb.Dataset(
        full_x, label=full_y, categorical_feature=cat_indices
    )

    best_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=study.best_trial.user_attrs.get(
            "best_iteration", cfg["model"]["max_iterations"]
        ),
    )

    return best_model, study


def save_model(model: lgb.Booster, path: str) -> None:
    """LightGBMモデルを保存する。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def load_model(path: str) -> lgb.Booster:
    """LightGBMモデルを読み込む。"""
    return lgb.Booster(model_file=path)
