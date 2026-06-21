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
    per_bet_cap: float | None = None,
) -> float:
    """フラクショナル・ケリー基準で 1 ベット分の賭け金 (円) を計算する。

    per_bet_cap 指定時、Kelly + max_bet_fraction の結果に対し min(per_bet_cap) を適用。
    """
    kf = kelly_fraction(prob, odds)
    if kf <= 0:
        return 0.0

    bet = bankroll * kf * fraction
    bet = min(bet, bankroll * max_bet_fraction)
    if per_bet_cap is not None:
        bet = min(bet, per_bet_cap)

    bet = int(bet // 100) * 100
    if bet < min_bet:
        if kf > 0:
            bet = min_bet
        else:
            bet = 0.0
    return float(bet)


def allocate_per_race_cap(amounts, per_race_cap, min_bet=100.0):
    """合計が per_race_cap 以下に収まるよう比例縮小 (100 円単位)。"""
    arr = np.asarray(amounts, dtype=float)
    total = arr.sum()
    if total <= per_race_cap or total <= 0:
        return arr.tolist()
    scale = per_race_cap / total
    scaled = arr * scale
    rounded = (np.floor(scaled / min_bet) * min_bet).astype(float)
    leftover = per_race_cap - rounded.sum()
    if leftover >= min_bet:
        idx = int(np.argmax(arr))
        rounded[idx] += int(leftover // min_bet) * min_bet
    return rounded.tolist()


def allocate_by_probability(probs, per_race_cap, min_bet=100.0, min_prob=0.30):
    """確率 ^ 2 重みで per_race_cap を按分 (オッズなし時のフォールバック)。"""
    arr = np.asarray(probs, dtype=float)
    mask = arr >= min_prob
    out = np.zeros_like(arr, dtype=float)
    if not mask.any():
        return out.tolist()
    w = arr[mask] ** 2
    w = w / w.sum()
    raw = w * per_race_cap
    rounded = (np.floor(raw / min_bet) * min_bet).astype(float)
    rounded[rounded < min_bet] = 0.0
    leftover = per_race_cap - rounded.sum()
    if leftover >= min_bet:
        sub_idx = int(np.argmax(arr[mask]))
        rounded[sub_idx] += int(leftover // min_bet) * min_bet
    out[mask] = rounded
    return out.tolist()


def size_bets_per_race(
    probs, odds, bankroll, per_race_cap,
    fraction=0.25, max_bet_fraction=0.05, min_bet=100.0, min_prob=0.30,
):
    """1 レース分の馬全頭への賭け金 (円) を割り当てる。

    オッズ取得済み → 各馬 Kelly → 合計 per_race_cap 内に収める
    オッズ未取得  → 確率重み付き擬似ケリー
    """
    p = np.asarray(probs, dtype=float)
    if odds is None:
        o = np.full_like(p, 0.0)
    else:
        o = np.asarray(odds, dtype=float)
    assert len(p) == len(o)
    raw = []
    have_any_odds = False
    for prob, oo in zip(p, o):
        if not np.isfinite(prob) or prob <= 0 or not np.isfinite(oo) or oo <= 0:
            raw.append(0.0)
            continue
        have_any_odds = True
        raw.append(compute_bet_amount(
            prob=float(prob), odds=float(oo), bankroll=bankroll,
            fraction=fraction, max_bet_fraction=max_bet_fraction,
            min_bet=min_bet, per_bet_cap=per_race_cap,
        ))
    if not have_any_odds:
        return allocate_by_probability(p, per_race_cap, min_bet=min_bet, min_prob=min_prob)
    return allocate_per_race_cap(raw, per_race_cap, min_bet=min_bet)


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
    """ティアベースのサイジング(閾値方式)で賭け金を計算する。

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
        kelly_keys = {"fraction", "max_bet_fraction", "min_bet", "per_bet_cap"}
        kelly_kwargs = {k: v for k, v in kwargs.items() if k in kelly_keys}
        return compute_bet_amount(prob, odds, bankroll, **kelly_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'tier' or 'kelly'.")
