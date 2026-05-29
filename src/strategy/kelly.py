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
    """フラクショナル・ケリー基準を用いて1頭分の賭け金 (円) を計算する。

    Args:
        prob: 勝率(または複勝率)の推定値
        odds: 同じ券種のオッズ(小数倍率)
        bankroll: 現在のバンクロール (円)
        fraction: ケリー倍率(0.25 = クォーターケリー)
        max_bet_fraction: バンクロールに対する 1 ベットの最大比率
        min_bet: 最小賭け金 (円, JRA は 100)
        per_bet_cap: 1ベット上限 (円)。指定時 max_bet_fraction と min で制限される。

    Returns:
        100 円単位に丸めた賭け金 (円)。
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


def allocate_per_race_cap(
    amounts: list[float] | np.ndarray,
    per_race_cap: float,
    min_bet: float = 100.0,
) -> list[float]:
    """1レース合計が ``per_race_cap`` を超える場合に按分で縮める。

    各馬の元の amounts(比率) を保ち、合計 = per_race_cap となるように 100 円単位で再分配する。
    すべて 0 ならそのまま返す。100円単位丸めの誤差は最大金額に寄せる。

    Args:
        amounts: 元の賭け金 (円)。長さ = 候補馬数。
        per_race_cap: 1レース合計上限 (円)。
        min_bet: 最小単位 (100円)。

    Returns:
        合計 ≤ per_race_cap を満たす新しい amounts (list[float])。
    """
    arr = np.asarray(amounts, dtype=float)
    total = arr.sum()
    if total <= per_race_cap or total <= 0:
        return arr.tolist()

    # 比率を保って scale down
    scale = per_race_cap / total
    scaled = arr * scale

    # 100 円単位に floor
    rounded = (np.floor(scaled / min_bet) * min_bet).astype(float)
    # 上限超過していないか念のため
    leftover = per_race_cap - rounded.sum()
    # leftover を最大スコアの馬に追加 (まだ余裕がある場合)
    if leftover >= min_bet:
        idx = int(np.argmax(arr))
        rounded[idx] += int(leftover // min_bet) * min_bet
    return rounded.tolist()


def allocate_by_probability(
    probs: list[float] | np.ndarray,
    per_race_cap: float,
    min_bet: float = 100.0,
    min_prob: float = 0.30,
) -> list[float]:
    """オッズが取れない場面で、確率に比例して per_race_cap を按分する。

    ``probs >= min_prob`` の馬だけを対象とし、確率比で per_race_cap を分配して
    100 円単位に丸めた賭け金を返す。他の馬は 0。

    高確率馬ほど厚く配分される擬似ケリーとして機能する。
    """
    arr = np.asarray(probs, dtype=float)
    mask = arr >= min_prob
    out = np.zeros_like(arr, dtype=float)
    if not mask.any():
        return out.tolist()

    w = arr[mask]
    # 重みは「確率^2」にして自信ある馬をさらに厚く (シャープな分配)
    w = w * w
    w = w / w.sum()
    raw = w * per_race_cap
    rounded = (np.floor(raw / min_bet) * min_bet).astype(float)
    # min_bet 未満の馬を 0 に
    rounded[rounded < min_bet] = 0.0

    leftover = per_race_cap - rounded.sum()
    if leftover >= min_bet:
        # 最高確率の馬に追加
        sub_idx = int(np.argmax(arr[mask]))
        real_idx = int(np.where(mask)[0][sub_idx])
        # まず該当行の合計が cap 内に収まる範囲で追加
        rounded[np.where(mask)[0][sub_idx]] += int(leftover // min_bet) * min_bet
    out[mask] = rounded
    return out.tolist()


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
        賭け金(float, 円)。

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


def size_bets_per_race(
    probs: list[float] | np.ndarray,
    odds: list[float] | np.ndarray | None,
    bankroll: float,
    per_race_cap: float,
    fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_bet: float = 100.0,
    min_prob: float = 0.30,
) -> list[float]:
    """1レース分の候補馬に対し、ケリー基準 + per_race_cap で実額(円)を割り当てる。

    流れ:
        1. オッズが取得済みの馬は ``compute_bet_amount`` でケリー賭け金を計算
        2. オッズ無し or 0 の馬はゼロ
        3. 全部オッズ無しなら ``allocate_by_probability`` で確率重み付け按分
        4. 合計が ``per_race_cap`` を超えていれば比例縮小

    Returns:
        各馬に対する賭け金 (円, 100 円単位)。長さ = len(probs)。
    """
    p = np.asarray(probs, dtype=float)
    if odds is None:
        o = np.full_like(p, fill_value=0.0)
    else:
        o = np.asarray(odds, dtype=float)
    assert len(p) == len(o)

    raw: list[float] = []
    have_any_odds = False
    for prob, oo in zip(p, o):
        if not np.isfinite(prob) or prob <= 0 or not np.isfinite(oo) or oo <= 0:
            raw.append(0.0)
            continue
        have_any_odds = True
        raw.append(
            compute_bet_amount(
                prob=float(prob), odds=float(oo), bankroll=bankroll,
                fraction=fraction, max_bet_fraction=max_bet_fraction,
                min_bet=min_bet, per_bet_cap=per_race_cap,
            )
        )

    if not have_any_odds:
        # オッズなし → 確率比按分のフォールバック
        return allocate_by_probability(p, per_race_cap=per_race_cap, min_bet=min_bet, min_prob=min_prob)

    return allocate_per_race_cap(raw, per_race_cap=per_race_cap, min_bet=min_bet)
