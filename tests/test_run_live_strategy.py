"""run_live の投票判定ロジック (API 非依存の純粋部分) のユニットテスト。

実行: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

# Anaconda 環境では site-packages に同名の `scripts` パッケージがあり
# `from scripts.run_live import ...` が失敗するため、scripts/ を直接パスに通す
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from run_live import (
    COMPETITION_DAYS,
    DEFAULT_LIVE_STRATEGY,
    compute_big_amount,
    decide_bet,
)


def make_strat(**over):
    strat = dict(DEFAULT_LIVE_STRATEGY)
    strat.update(over)
    return strat


def make_race(probs, horse_nums=None):
    n = len(probs)
    return pd.DataFrame({
        "horse_num": horse_nums or list(range(1, n + 1)),
        "win_pred_prob": probs,
    })


def make_odds(mapping):
    """{馬番: オッズ} → win_odds_df"""
    return pd.DataFrame({
        "comb": [str(k).zfill(2) for k in mapping],
        "odds": list(mapping.values()),
    })


class TestComputeBigAmount(unittest.TestCase):
    def test_initial_day_targets_landing(self):
        """初日: 残り 9 日で目標 51 万に着地する大口額が出ること。"""
        strat = make_strat()
        state = {"total_wagered": 0}
        amt = compute_big_amount(strat, state, COMPETITION_DAYS[0])
        # need = 510000 - 9*36*100 = 477,600 / 大口機会 9*36*0.06 ≈ 19.4 → ~24,600
        self.assertGreaterEqual(amt, 20000)
        self.assertLessEqual(amt, 30000)
        self.assertEqual(amt % 100, 0)

    def test_target_reached_falls_to_min(self):
        """目標達成済みなら大口は下限に落ちる (追加損失の抑制)。"""
        strat = make_strat()
        state = {"total_wagered": 520000}
        amt = compute_big_amount(strat, state, COMPETITION_DAYS[-1])
        self.assertEqual(amt, int(strat["big_amount_min"]))

    def test_final_day_behind_schedule_hits_cap(self):
        """最終日に大幅未達なら上限にクランプされる。"""
        strat = make_strat()
        state = {"total_wagered": 100000}
        amt = compute_big_amount(strat, state, COMPETITION_DAYS[-1])
        self.assertEqual(amt, int(strat["big_amount_max"]))


class TestDecideBetFavoriteConcentration(unittest.TestCase):
    def test_big_bet_on_heavy_favorite(self):
        """top-1 のオッズが 1.5 以下なら大口 (fav_big)。"""
        strat = make_strat(mode="favorite_concentration")
        race = make_race([0.45, 0.20, 0.10])
        odds = make_odds({1: 1.3, 2: 4.5, 3: 9.0})
        d = decide_bet(race, odds, strat, big_amount=25000)
        self.assertEqual(d["bet_kind"], "fav_big")
        self.assertEqual(d["horse_num"], 1)
        self.assertEqual(d["amount"], 25000)

    def test_min_bet_when_odds_above_threshold(self):
        """top-1 のオッズが 1.5 超なら最低額 (min)。"""
        strat = make_strat(mode="favorite_concentration")
        race = make_race([0.30, 0.25, 0.10])
        odds = make_odds({1: 2.8, 2: 3.1, 3: 8.0})
        d = decide_bet(race, odds, strat, big_amount=25000)
        self.assertEqual(d["bet_kind"], "min")
        self.assertEqual(d["amount"], int(strat["min_bet_per_race"]))

    def test_min_bet_when_odds_missing(self):
        """オッズ取得失敗時も最低額投票は行う (96 レース制約を守る)。"""
        strat = make_strat(mode="favorite_concentration")
        race = make_race([0.30, 0.25])
        d = decide_bet(race, pd.DataFrame(columns=["comb", "odds"]), strat,
                       big_amount=25000)
        self.assertEqual(d["bet_kind"], "min")
        self.assertEqual(d["horse_num"], 1)

    def test_top1_selected_by_prob_not_order(self):
        """top-1 は行順ではなく win_pred_prob 最大で選ぶ。"""
        strat = make_strat(mode="favorite_concentration")
        race = make_race([0.10, 0.40, 0.20], horse_nums=[3, 7, 12])
        odds = make_odds({7: 1.2})
        d = decide_bet(race, odds, strat, big_amount=10000)
        self.assertEqual(d["horse_num"], 7)
        self.assertEqual(d["bet_kind"], "fav_big")

    def test_no_big_bet_without_budget(self):
        """big_amount=0 (目標達成後など) では大口を打たない。"""
        strat = make_strat(mode="favorite_concentration")
        race = make_race([0.45])
        odds = make_odds({1: 1.2})
        d = decide_bet(race, odds, strat, big_amount=0)
        self.assertEqual(d["bet_kind"], "min")

    def test_empty_race_returns_none(self):
        strat = make_strat()
        self.assertIsNone(decide_bet(pd.DataFrame(), pd.DataFrame(), strat))


if __name__ == "__main__":
    unittest.main()
