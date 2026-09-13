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
from itertools import combinations, combinations_with_replacement, permutations

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
    "place":       2,  # 複勝
    "waku_rensho": 3,  # 枠連
    "quinella":    4,  # 馬連
    "wide":        5,  # ワイド
    "exacta":      6,  # 馬単
    "trio":        7,  # 3連複
    "trifecta":    8,  # 3連単
}

# 控除率 (JRA 公表)
TAKEOUT_ALL = {
    "place":       0.20,
    "waku_rensho": 0.225,
    "quinella":    0.225,
    "wide":        0.225,
    "exacta":      0.25,
    "trio":        0.25,
    "trifecta":    0.275,
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



def joint_probs_place(win_probs: np.ndarray, top_n: int = 8) -> pd.DataFrame:
    """複勝: 各馬の rank<=3 に入る確率 = P(i∈{1st,2nd,3rd})。

    Harville 展開で P(i=1st) + P(i=2nd) + P(i=3rd)。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:min(top_n * 2, len(win_probs))]  # 分母用に多めに
    rows = []
    for i in top_idx[:n]:
        p_i = float(win_probs[i])
        # 1st
        p1 = p_i
        # 2nd: sum_j p_j * p_i / (1 - p_j)
        p2 = 0.0
        for j in range(len(win_probs)):
            if j == i: continue
            p_j = float(win_probs[j])
            denom = 1.0 - p_j
            if denom > 1e-9:
                p2 += p_j * p_i / denom
        # 3rd: sum_{j,k} p_j * p_k/(1-p_j) * p_i/(1-p_j-p_k)
        p3 = 0.0
        for j in range(len(win_probs)):
            if j == i: continue
            p_j = float(win_probs[j])
            d_j = 1.0 - p_j
            if d_j <= 1e-9: continue
            for k in range(len(win_probs)):
                if k == i or k == j: continue
                p_k = float(win_probs[k])
                d_jk = 1.0 - p_j - p_k
                if d_jk <= 1e-9: continue
                p3 += p_j * (p_k / d_j) * (p_i / d_jk)
        rows.append((int(i), p1 + p2 + p3))
    return pd.DataFrame(rows, columns=["h1", "joint_prob"])


def joint_probs_wide(win_probs: np.ndarray, top_n: int = 6) -> pd.DataFrame:
    """ワイド: 2頭 {i,j} が両方とも rank<=3 の確率。

    6 通りの順列 (i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i) を
    k で marginalize して合計。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    n = min(top_n, len(win_probs))
    top_idx = np.argsort(win_probs)[::-1][:n]
    rows = []
    for a, b in combinations(sorted(top_idx), 2):
        prob = 0.0
        # 3 着目の候補は全馬 (a, b 以外)
        for k in range(len(win_probs)):
            if k == a or k == b: continue
            # 6 順列を Harville で
            for i, j, l in [(a, b, k), (a, k, b), (b, a, k), (b, k, a), (k, a, b), (k, b, a)]:
                prob += harville_probability(win_probs, i, j, l)
        rows.append((int(a), int(b), prob))
    return pd.DataFrame(rows, columns=["h1", "h2", "joint_prob"])


def joint_probs_waku_rensho(
    win_probs: np.ndarray, waku_nums: np.ndarray, max_waku: int = 8,
) -> pd.DataFrame:
    """枠連: 2枠が rank<=2 に入る確率 (順不同、同枠 = ゾロ目も可)。

    P(枠A=1st ∧ 枠B=2nd) + P(枠B=1st ∧ 枠A=2nd) を、枠内の全馬で marginalize。
    """
    win_probs = np.asarray(win_probs, dtype=np.float64)
    waku_nums = np.asarray(waku_nums, dtype=np.int64)
    unique_wakus = sorted(set(int(w) for w in waku_nums if w > 0))
    rows = []
    for a, b in combinations_with_replacement(unique_wakus, 2):
        prob = 0.0
        # 枠 a と枠 b の馬の全ペアで sum
        idx_a = [i for i in range(len(win_probs)) if int(waku_nums[i]) == a]
        idx_b = [i for i in range(len(win_probs)) if int(waku_nums[i]) == b]
        for i in idx_a:
            for j in idx_b:
                if i == j: continue  # 同一馬は除外 (ゾロ目枠内でも別馬を選ぶ)
                p_i = float(win_probs[i]); p_j = float(win_probs[j])
                # P(i=1st, j=2nd) + P(j=1st, i=2nd)
                d_i = 1.0 - p_i
                d_j = 1.0 - p_j
                if d_i > 1e-9:
                    prob += p_i * p_j / d_i
                if d_j > 1e-9 and a != b:  # a==b の場合は 1 通り分だけ
                    prob += p_j * p_i / d_j
        rows.append((int(a), int(b), prob))
    return pd.DataFrame(rows, columns=["h1", "h2", "joint_prob"])


def select_all_bets(
    horse_nums: np.ndarray,
    win_probs: np.ndarray,
    odds_df: pd.DataFrame,
    strat: dict,
    waku_nums: np.ndarray | None = None,
) -> list[MultiBetCandidate]:
    """全 7 券種 (複勝〜三連単) を横断的に評価し、利益フィルタで買い目を選定。

    strat 例:
        {
            "all_bets_enabled": True,
            "all_bets_amount": 1000,               # 1 買い目 pt
            "all_bets_min_profit_if_hit": 100000,  # 的中時利益 = amount×(odds-1) の閾値
            "all_bets_min_prob": {                 # 種別ごとの最小 joint prob (超大穴を除外)
                "place": 0.10, "waku_rensho": 0.02, "quinella": 0.005,
                "wide": 0.02, "exacta": 0.002, "trio": 0.001, "trifecta": 0.0005,
            },
            "all_bets_top_n": 6,                   # 上位 N 頭を考慮
            "all_bets_max_bets_per_race": 20,      # 1 レース最大点数
            "all_bets_use_padded_bet_id": True,    # 9/6 の kaime エラー回避
        }
    """
    if odds_df is None or odds_df.empty:
        return []
    # 賭け金: scalar (全券種一律) or dict (券種別)。dict の _default は既定値
    _amount_cfg = strat.get("all_bets_amount", 1000)
    if isinstance(_amount_cfg, dict):
        _amount_dict = {k: int(v) for k, v in _amount_cfg.items()}
        _amount_default = int(_amount_dict.get("_default", 1000))
    else:
        _amount_dict = {}
        _amount_default = int(_amount_cfg)
    def _amount_for(bt: str) -> int:
        return _amount_dict.get(bt, _amount_default)
    # 利益フィルタ: scalar なら全券種一律、dict なら券種別に上書き
    _min_profit_cfg = strat.get("all_bets_min_profit_if_hit", 100000)
    if isinstance(_min_profit_cfg, dict):
        _profit_dict = {k: float(v) for k, v in _min_profit_cfg.items()}
        _profit_default = float(_profit_dict.get("_default", 100000))
    else:
        _profit_dict = {}
        _profit_default = float(_min_profit_cfg)
    def _min_odds_for(bt: str) -> float:
        # amount × (odds - 1) >= min_profit → odds >= min_profit/amount + 1
        mp = _profit_dict.get(bt, _profit_default)
        return mp / _amount_for(bt) + 1.0
    min_probs = strat.get("all_bets_min_prob", {})
    top_n = int(strat.get("all_bets_top_n", 6))
    max_bets = int(strat.get("all_bets_max_bets_per_race", 20))

    n_horses = len(horse_nums)
    if n_horses < 2:
        return []
    idx_to_hnum = {i: int(horse_nums[i]) for i in range(n_horses)}
    hnum_to_idx = {int(horse_nums[i]): i for i in range(n_horses)}

    # odds_df: (comb, odds_type, odds) を dict に
    def _split_comb(comb: str, k: int) -> tuple[int, ...]:
        c = str(comb).strip()
        if "-" in c:
            return tuple(int(x) for x in c.split("-"))
        if len(c) == 2 * k:
            return tuple(int(c[i:i+2]) for i in range(0, len(c), 2))
        return tuple()

    odds_map: dict[tuple[str, tuple[int, ...]], float] = {}
    type_map = {2: "place", 3: "waku_rensho", 4: "quinella", 5: "wide",
                6: "exacta", 7: "trio", 8: "trifecta"}
    key_size = {"place": 1, "waku_rensho": 2, "quinella": 2, "wide": 2,
                "exacta": 2, "trio": 3, "trifecta": 3}
    for _, row in odds_df.iterrows():
        try:
            otype = int(row["odds_type"])
        except (ValueError, TypeError, KeyError):
            continue
        bt = type_map.get(otype)
        if bt is None: continue
        k = key_size[bt]
        horses = _split_comb(row.get("comb"), k)
        if len(horses) != k: continue
        try:
            o = float(row["odds"])
        except (ValueError, TypeError):
            continue
        if o < _min_odds_for(bt): continue  # 利益フィルタ (券種別)
        # 順序不問系は昇順キー
        if bt in ("place", "waku_rensho", "quinella", "wide", "trio"):
            key = tuple(sorted(horses))
        else:
            key = tuple(horses)
        odds_map[(bt, key)] = o

    if not odds_map:
        return []

    # 各券種の joint prob を計算
    candidates: list[MultiBetCandidate] = []

    # 複勝
    if any(k[0] == "place" for k in odds_map):
        pdf = joint_probs_place(win_probs, top_n=top_n)
        for _, r in pdf.iterrows():
            i = int(r["h1"])
            hn = idx_to_hnum[i]
            key = (hn,)
            o = odds_map.get(("place", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("place", 0.10): continue
            candidates.append(MultiBetCandidate(
                bet_type="place", horses=(hn,),
                joint_prob=prob, odds=o, ev=prob * o, amount=amount,
            ))

    # ワイド
    if any(k[0] == "wide" for k in odds_map):
        wdf = joint_probs_wide(win_probs, top_n=top_n)
        for _, r in wdf.iterrows():
            i, j = int(r["h1"]), int(r["h2"])
            key = tuple(sorted([idx_to_hnum[i], idx_to_hnum[j]]))
            o = odds_map.get(("wide", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("wide", 0.02): continue
            candidates.append(MultiBetCandidate(
                bet_type="wide", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("wide"),
            ))

    # 枠連
    if any(k[0] == "waku_rensho" for k in odds_map) and waku_nums is not None:
        try:
            wdf = joint_probs_waku_rensho(win_probs, waku_nums)
        except Exception:
            wdf = pd.DataFrame(columns=["h1", "h2", "joint_prob"])
        for _, r in wdf.iterrows():
            a, b = int(r["h1"]), int(r["h2"])
            key = tuple(sorted([a, b]))
            o = odds_map.get(("waku_rensho", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("waku_rensho", 0.02): continue
            candidates.append(MultiBetCandidate(
                bet_type="waku_rensho", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("waku_rensho"),
            ))

    # 馬連 / 馬単 / 三連複 / 三連単 (既存関数を使い回し)
    if any(k[0] == "quinella" for k in odds_map):
        qdf = joint_probs_quinella(win_probs, top_n=top_n)
        for _, r in qdf.iterrows():
            i, j = int(r["h1"]), int(r["h2"])
            key = tuple(sorted([idx_to_hnum[i], idx_to_hnum[j]]))
            o = odds_map.get(("quinella", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("quinella", 0.005): continue
            candidates.append(MultiBetCandidate(
                bet_type="quinella", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("quinella"),
            ))
    if any(k[0] == "exacta" for k in odds_map):
        edf = joint_probs_exacta(win_probs, top_n=top_n)
        for _, r in edf.iterrows():
            i, j = int(r["h1"]), int(r["h2"])
            key = (idx_to_hnum[i], idx_to_hnum[j])
            o = odds_map.get(("exacta", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("exacta", 0.002): continue
            candidates.append(MultiBetCandidate(
                bet_type="exacta", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("exacta"),
            ))
    if any(k[0] == "trio" for k in odds_map):
        tdf = joint_probs_trio(win_probs, top_n=top_n)
        for _, r in tdf.iterrows():
            i, j, k_i = int(r["h1"]), int(r["h2"]), int(r["h3"])
            key = tuple(sorted([idx_to_hnum[i], idx_to_hnum[j], idx_to_hnum[k_i]]))
            o = odds_map.get(("trio", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("trio", 0.001): continue
            candidates.append(MultiBetCandidate(
                bet_type="trio", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("trio"),
            ))
    if any(k[0] == "trifecta" for k in odds_map):
        tfdf = joint_probs_trifecta(win_probs, top_n=top_n)
        for _, r in tfdf.iterrows():
            i, j, k_i = int(r["h1"]), int(r["h2"]), int(r["h3"])
            key = (idx_to_hnum[i], idx_to_hnum[j], idx_to_hnum[k_i])
            o = odds_map.get(("trifecta", key))
            if o is None: continue
            prob = float(r["joint_prob"])
            if prob < min_probs.get("trifecta", 0.0005): continue
            candidates.append(MultiBetCandidate(
                bet_type="trifecta", horses=key,
                joint_prob=prob, odds=o, ev=prob * o, amount=_amount_for("trifecta"),
            ))

    # EV 順に上位を返す (件数制限)
    candidates.sort(key=lambda c: c.ev, reverse=True)
    return candidates[:max_bets]
