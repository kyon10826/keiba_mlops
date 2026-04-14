"""賭け金計算のためのフラクショナル・ケリー基準およびティアベースのサイジング。"""

from __future__ import annotations

import numpy as np


def kelly_fraction(prob: float, odds: float) -> float:
    """フルケリー比率を計算する。

    Args:
        prob: 勝利確率の推定値
        odds: オッズ(単位賭け金あたりの払戻。例: 3.0 は3倍のリターン)

    Returns:
        ケリー比率(期待値がマイナスの場合は負の値となることがある)
    """
    b = odds - 1.0  # 正味オッズ
    q = 1.0 - prob
    if b <= 0:
        return 0.0
    return (prob * b - q) / b


def compute_bet_amount(
    prob: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
) -> float:
    """フラクショナル・ケリー基準を用いて賭け金を計算する。

    Args:
        prob: 複勝(1-3着)確率の推定値
        odds: 複勝オッズの推定値(小数)
        bankroll: 現在のバンクロール
        fraction: ケリー倍率(0.25 = クォーターケリー)
        max_bet_fraction: バンクロールに対する最大賭け金比率
        min_bet: 最小賭け金(プラットフォームの最低額)

    Returns:
        100単位に丸めた賭け金(ポイント単位)
    """
    kf = kelly_fraction(prob, odds)

    if kf <= 0:
        return 0.0

    # フラクショナル・ケリーを適用
    bet = bankroll * kf * fraction

    # バンクロールの最大比率で上限を設ける
    max_bet = bankroll * max_bet_fraction
    bet = min(bet, max_bet)

    # 100単位に丸める
    bet = int(bet // 100) * 100

    # 最小値を適用
    if bet < min_bet:
        # ケリーが賭けると判断したが最小値未満の場合は最小値を賭ける
        # (期待値が正の場合に限る)
        if kf > 0:
            bet = min_bet
        else:
            bet = 0.0

    return float(bet)


def compute_bet_amounts_batch(
    probs: np.ndarray,
    odds: np.ndarray,
    bankroll: float,
    fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
) -> np.ndarray:
    """compute_bet_amount のベクトル化版。"""
    amounts = np.array([
        compute_bet_amount(p, o, bankroll, fraction, max_bet_fraction, min_bet)
        for p, o in zip(probs, odds)
    ])
    return amounts


def compute_tier_bet_amount(
    prob: float,
    tier_low_threshold: float = 0.3,
    tier_mid_threshold: float = 0.4,
    tier_high_threshold: float = 0.5,
    tier_low_amount: float = 100.0,
    tier_mid_amount: float = 300.0,
    tier_high_amount: float = 500.0,
) -> float:
    """ティアベースのサイジング(Frieren方式)で賭け金を計算する。

    予測確率のティアに基づいて賭け金を割り当てる:
    - prob >= tier_high_threshold → 強気買い (tier_high_amount)
    - prob >= tier_mid_threshold  → 通常買い (tier_mid_amount)
    - prob >= tier_low_threshold  → 小額買い (tier_low_amount)
    - prob < tier_low_threshold   → 見送り (0)

    オッズは不要で、モデルの予測確率のみを使用する。

    Returns:
        賭け金(float)。最低閾値未満の場合は 0.0。
    """
    if prob >= tier_high_threshold:
        amount = tier_high_amount
    elif prob >= tier_mid_threshold:
        amount = tier_mid_amount
    elif prob >= tier_low_threshold:
        amount = tier_low_amount
    else:
        return 0.0

    # 100単位に丸める
    return float(int(amount // 100) * 100)


def compute_tier_bet_amounts_batch(
    probs: np.ndarray,
    **tier_kwargs,
) -> np.ndarray:
    """compute_tier_bet_amount のベクトル化版。"""
    amounts = np.array([
        compute_tier_bet_amount(p, **tier_kwargs)
        for p in probs
    ])
    return amounts


def compute_bet_amount_dispatch(
    prob: float,
    odds: float | None = None,
    bankroll: float | None = None,
    method: str = "tier",
    **kwargs,
) -> float:
    """method に応じてティアまたはケリーのサイジングにディスパッチする。

    Args:
        prob: 勝利確率の推定値。
        odds: オッズ(kelly メソッドで必須)。
        bankroll: 現在のバンクロール(kelly メソッドで必須)。
        method: "tier" または "kelly"。
        **kwargs: 各下位関数に渡す追加のキーワード引数。

    Returns:
        賭け金(float)。

    Raises:
        ValueError: method が不明な場合、または kelly 選択時に odds/bankroll が無い場合。
    """
    if method == "tier":
        tier_keys = {
            "tier_low_threshold", "tier_mid_threshold", "tier_high_threshold",
            "tier_low_amount", "tier_mid_amount", "tier_high_amount",
        }
        tier_kwargs = {k: v for k, v in kwargs.items() if k in tier_keys}
        return compute_tier_bet_amount(prob, **tier_kwargs)
    elif method == "kelly":
        if odds is None or bankroll is None:
            raise ValueError("kelly method requires both odds and bankroll")
        kelly_keys = {"fraction", "max_bet_fraction", "min_bet"}
        kelly_kwargs = {k: v for k, v in kwargs.items() if k in kelly_keys}
        return compute_bet_amount(prob, odds, bankroll, **kelly_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'tier' or 'kelly'.")
