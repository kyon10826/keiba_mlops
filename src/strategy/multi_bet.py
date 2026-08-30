"""多重ベット (馬連 / 馬単 / 三連複 / 三連単) の EV 選定。

Harville モデルで単勝確率から joint 確率を算出し、5 分前オッズと結合して
EV = joint_prob × odds を計算、閾値以上の買い目を返す。

用途: 単勝でモデル top-1 が固いレースに、周辺馬との組合せでバックエンド
狙い。控除率 (馬連 22.5% / 馬単 25% / 三連複 25% / 三連単 27.5%) を
考慮しても EV が正になる買い目だけを抽出する。

期待値の観点:
    平均的には全券種で EV < 1.0 だが、モデルの top-3 が的中する確率が
    Harville 予測より高いレース (=モデルが得意なレース) では EV > 1.0
    の組合せが少数存在する。それだけを拾って積む戦略。

    ただし三連単の平均配当 100-500 倍・的中率 1% 前後という高分散なので、
    実現収支の分散は非常に大きい (1 日で +30 万や -3 万を行き来しうる)。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from src.strategy.harville import harville_probability


# 券種ごとの控除率 (JRA 公表)
TAKEOUT = {
    "quinella":  0.225,   # 馬連
    "exacta":    0.25,    # 馬単
    "trio":      0.25,    # 三連複
    "trifecta":  0.275,   # 三連単
}

# 券種番号 (大会 API の odds_type と bet_id b{N}_c0_...)
# データ仕様書: 単勝1, 複勝2, 枠連3, 馬連4, ワイド5, 馬単6, 3連複7, 3連単8
BET_TYPE_CODE = {
    "quinella": 4,
    "exacta":   6,
    "trio":     7,
    "trifecta": 8,
}


@dataclass
class MultiBetCandidate:
    """1 レース分の 1 買い目。"""
    bet_type: str            # "trio" | "trifecta" | "exacta" | "quinella"
    horses: tuple[int, ...]  # 馬番のタプル (順序あり: trifecta/exacta / 順序なし: trio/quinella)
    joint_prob: float        # Harville 由来の的中確率
    odds: float              # 5 分前オッズ
    ev: float                # joint_prob × odds
    amount: int              # 賭け金 (100 pt 単位)

    @property
    def bet_id(self) -> str:
        """大会投票 API の bet_id 形式。

        単勝サンプル: 'b1_c0_9' (馬番そのまま、ゼロ埋めなし)
        推定形式: 'b{券種}_c0_{馬番をハイフン区切り}' 例 'b7_c0_1-2-3'
        ※ 実際の運営仕様と違う場合は初回投票で NG が返るので config で切替可能に。
        """
        code = BET_TYPE_CODE[self.bet_type]
        # 順序あり (trifecta/exacta): 着順どおり
        # 順序なし (trio/quinella): 馬番昇順
        if self.bet_type in ("trio", "quinella"):
            horses = tuple(sorted(self.horses))
        else:
            horses = self.horses
        return f"b{code}_c0_" + "-".join(str(h) for h in horses)

    @property
    def bet_id_padded(self) -> str:
        """代替形式: 大会仕様書 comb 例 '130710' に倣ったゼロ埋め。

        POST /bet で 'b7_c0_1-2-3' が NG なら 'b7_c0_010203' を試す。
        """
        code = BET_TYPE_CODE[self.bet_type]
        if self.bet_type in ("trio", "quinella"):
            horses = tuple(sorted(self.horses))
        else:
            horses = self.horses
        return f"b{code}_c0_" + "".join(str(h).zfill(2) for h in horses)


def joint_probs_trifecta(win_probs: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """三連単の (i,j,k) 順列と Harville 確率。"""
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:n]
    rows = []
    for i, j, k in permutations(top_idx, 3):
        rows.append((int(i), int(j), int(k), harville_probability(win_probs, i, j, k)))
    return pd.DataFrame(rows, columns=["h1", "h2", "h3", "joint_prob"])


def joint_probs_trio(win_probs: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """三連複の {i,j,k} 組合せ (順不同) と合計 Harville 確率。"""
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:n]
    rows = []
    for combo in combinations(sorted(top_idx), 3):
        prob = sum(harville_probability(win_probs, i, j, k)
                   for i, j, k in permutations(combo))
        rows.append((int(combo[0]), int(combo[1]), int(combo[2]), prob))
    return pd.DataFrame(rows, columns=["h1", "h2", "h3", "joint_prob"])


def joint_probs_exacta(win_probs: np.ndarray, top_n: int = 6) -> pd.DataFrame:
    """馬単の (i,j) 順列と Harville 確率 (2 頭のみ考慮)。"""
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:n]
    rows = []
    for i, j in permutations(top_idx, 2):
        p_i = float(win_probs[i]); p_j = float(win_probs[j])
        denom = 1.0 - p_i
        prob = p_i * (p_j / denom) if p_i > 0 and p_j > 0 and denom > 0 else 0.0
        rows.append((int(i), int(j), prob))
    return pd.DataFrame(rows, columns=["h1", "h2", "joint_prob"])


def joint_probs_quinella(win_probs: np.ndarray, top_n: int = 6) -> pd.DataFrame:
    """馬連の {i,j} 組合せ (順不同) と合計 Harville 確率。"""
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:n]
    rows = []
    for combo in combinations(sorted(top_idx), 2):
        i, j = combo
        p_i = float(win_probs[i]); p_j = float(win_probs[j])
        # 2 通り
        p_ij = p_i * (p_j / max(1e-12, 1 - p_i))
        p_ji = p_j * (p_i / max(1e-12, 1 - p_j))
        rows.append((int(i), int(j), p_ij + p_ji))
    return pd.DataFrame(rows, columns=["h1", "h2", "joint_prob"])


def _apply_takeout_boundary(ev: float, bet_type: str) -> bool:
    """控除率を超えて期待値プラスかどうか (単純基準)。

    平均返還率 = 1 - takeout。EV >= 1.0 は「均衡オッズなら±0」。
    控除率超過を狙うなら EV >= 1 / (1 - takeout)。
    例: 三連単 takeout 0.275 → EV >= 1.379 で厳格プラス EV。
    """
    threshold = 1.0 / (1.0 - TAKEOUT[bet_type])
    return ev >= threshold


def select_multi_bets(
    horse_nums: np.ndarray,        # 各馬の馬番 (1-18)
    win_probs: np.ndarray,         # 校正後単勝確率
    odds_df: pd.DataFrame,         # comb (str) と odds_type (int), odds (float)
    strat: dict,
) -> list[MultiBetCandidate]:
    """複数券種を横断的に評価し、EV 上位 N 点を返す。

    strat 例:
        {
            "multi_bet_types": ["trio", "trifecta"],
            "multi_top_n_trio": 4,          # 上位 4 頭で trio の全組合せを評価
            "multi_top_n_trifecta": 4,
            "multi_top_n_exacta": 5,
            "multi_top_n_quinella": 5,
            "multi_min_ev": 1.30,           # 控除率 25% を超える閾値 (三連複)
            "multi_min_prob_trio": 0.005,
            "multi_min_prob_trifecta": 0.002,
            "multi_min_prob_exacta": 0.01,
            "multi_min_prob_quinella": 0.02,
            "multi_max_odds": 500.0,        # 超大穴の確率推定は不安定
            "multi_amount_per_bet": 300,
            "multi_max_bets_per_race": 6,
            "multi_max_total_per_race": 3000,
        }
    """
    if odds_df is None or odds_df.empty:
        return []
    n_horses = len(horse_nums)
    if n_horses < 2:
        return []

    # comb (str) から馬番 tuple を復元
    def _split_comb(comb: str, k: int) -> tuple[int, ...]:
        c = str(comb).strip()
        # 大会 API 仕様: ゼロ埋め 2/4/6 桁
        if "-" in c:
            return tuple(int(x) for x in c.split("-"))
        if len(c) == 2 * k:
            return tuple(int(c[i:i+2]) for i in range(0, len(c), 2))
        return tuple()

    types = list(strat.get("multi_bet_types", ["trio", "trifecta"]))
    min_ev = float(strat.get("multi_min_ev", 1.30))
    max_odds = float(strat.get("multi_max_odds", 500.0))

    # インデックス → 馬番マップ
    idx_to_hnum = {i: int(horse_nums[i]) for i in range(n_horses)}
    hnum_to_odds: dict[tuple[str, tuple[int, ...]], float] = {}

    # odds_df を bet_type × horse-tuple → odds の辞書に変換
    for _, row in odds_df.iterrows():
        try:
            otype = int(row["odds_type"])
        except (ValueError, TypeError, KeyError):
            continue
        bet_type = {4: "quinella", 6: "exacta", 7: "trio", 8: "trifecta"}.get(otype)
        if bet_type not in types:
            continue
        k = 3 if bet_type in ("trio", "trifecta") else 2
        horses = _split_comb(row.get("comb"), k)
        if len(horses) != k:
            continue
        try:
            o = float(row["odds"])
        except (ValueError, TypeError):
            continue
        if o <= 0 or o > max_odds:
            continue
        # trio/quinella は昇順キー、trifecta/exacta は順序保持
        if bet_type in ("trio", "quinella"):
            key = tuple(sorted(horses))
        else:
            key = tuple(horses)
        hnum_to_odds[(bet_type, key)] = o

    all_cand: list[MultiBetCandidate] = []
    amount = int(strat.get("multi_amount_per_bet", 300))

    for bet_type in types:
        top_n = int(strat.get(f"multi_top_n_{bet_type}", 5))
        min_prob = float(strat.get(f"multi_min_prob_{bet_type}", 0.002))
        if bet_type == "trio":
            jp = joint_probs_trio(win_probs, top_n)
            k = 3
        elif bet_type == "trifecta":
            jp = joint_probs_trifecta(win_probs, top_n)
            k = 3
        elif bet_type == "exacta":
            jp = joint_probs_exacta(win_probs, top_n)
            k = 2
        elif bet_type == "quinella":
            jp = joint_probs_quinella(win_probs, top_n)
            k = 2
        else:
            continue

        cols = [f"h{i+1}" for i in range(k)]
        for _, r in jp.iterrows():
            if float(r["joint_prob"]) < min_prob:
                continue
            horses_idx = tuple(int(r[c]) for c in cols)
            horses = tuple(idx_to_hnum[i] for i in horses_idx)
            if bet_type in ("trio", "quinella"):
                key = tuple(sorted(horses))
            else:
                key = tuple(horses)
            o = hnum_to_odds.get((bet_type, key))
            if o is None:
                continue
            prob = float(r["joint_prob"])
            ev = prob * o
            if ev < min_ev:
                continue
            all_cand.append(MultiBetCandidate(
                bet_type=bet_type,
                horses=horses,
                joint_prob=prob,
                odds=o,
                ev=ev,
                amount=amount,
            ))

    # EV 上位順に並べ、点数と合計金額の上限で絞る
    all_cand.sort(key=lambda c: c.ev, reverse=True)
    max_bets = int(strat.get("multi_max_bets_per_race", 6))
    max_total = int(strat.get("multi_max_total_per_race", 3000))
    selected: list[MultiBetCandidate] = []
    total = 0
    for c in all_cand:
        if len(selected) >= max_bets:
            break
        if total + c.amount > max_total:
            continue
        selected.append(c)
        total += c.amount
    return selected
