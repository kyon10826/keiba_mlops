"""ホールドアウトデータを用いた確率キャリブレーション。

元のノートブックで学習データ上でキャリブレーションを行っていた
データリークの問題を修正する。
"""

from __future__ import annotations

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class HoldoutCalibrator:
    """ホールドアウト分割で学習するIsotonic回帰キャリブレータ。

    使い方:
        1. 検証データを2つに分割する。
        2. 前半の生予測と真のラベルでキャリブレータを学習する。
        3. 後半で評価する。
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """ホールドアウト予測でキャリブレータを学習する。"""
        self.calibrator.fit(raw_probs, y_true)

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """生の確率をキャリブレーションする。"""
        return self.calibrator.predict(raw_probs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.calibrator, f)

    @classmethod
    def load(cls, path: str) -> "HoldoutCalibrator":
        obj = cls()
        with open(path, "rb") as f:
            obj.calibrator = pickle.load(f)
        return obj


def calibrate_model(
    model: lgb.Booster,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    holdout_fraction: float = 0.5,
    seed: int = 17,
) -> tuple[HoldoutCalibrator, pd.DataFrame, pd.Series]:
    """検証データのホールドアウト分割でモデルをキャリブレーションする。

    時系列データでのデータリークを防ぐため、ランダムな並び替えではなく
    時間順(前半/後半)の分割を使用する。検証データのうち最初の
    ``holdout_fraction`` (より古い期間)をキャリブレーション用に、
    残り(より新しい期間)を評価用に使用する。
    呼び出し側は、時系列順にソートされた検証データを渡すことが
    期待される。

    Args:
        model: 学習済みのLightGBMモデル
        valid_x: 検証用特徴量(時系列順にソート済み)
        valid_y: 検証用ラベル(時系列順にソート済み)
        holdout_fraction: キャリブレーションに使う割合(残りは評価用)
        seed: 乱数シード(未使用、API互換のため残している)

    Returns:
        (calibrator, eval_x, eval_y) — キャリブレータと評価用分割
    """
    n = len(valid_x)
    split = int(n * holdout_fraction)

    cal_x = valid_x.iloc[:split]
    cal_y = valid_y.iloc[:split].values

    eval_x = valid_x.iloc[split:]
    eval_y = valid_y.iloc[split:]

    raw_probs = model.predict(cal_x)

    calibrator = HoldoutCalibrator()
    calibrator.fit(raw_probs, cal_y)

    return calibrator, eval_x, eval_y
