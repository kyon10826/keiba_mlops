"""穴狙いサブベット (decide_anaba_bet) のユニットテスト。

実行: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from run_live import DEFAULT_LIVE_STRATEGY, decide_anaba_bet


def strat(**over):
    s = dict(DEFAULT_LIVE_STRATEGY)
    s.update(over)
    return s


def race(probs, horse_nums=None):
    n = len(probs)
    return pd.DataFrame({
        "horse_num": horse_nums or list(range(1, n + 1)),
        "win_pred_prob": probs,
    })


def odds(mapping):
    return pd.DataFrame({
        "comb": [str(k).zfill(2) for k in mapping],
        "odds": list(mapping.values()),
    })


class TestDecideAnabaBet(unittest.TestCase):
    def test_picks_qualifying_longshot(self):
        """10 倍以上 & 勝率閾値以上の馬を選ぶ (テストは min_prob=0.10, max_odds=30 で厳格化)。"""
        s = strat(anaba_min_prob=0.10, anaba_max_odds=30.0)
        r = race([0.35, 0.15, 0.05])  # 馬 1=本命, 馬 2=穴候補, 馬 3=切 (prob<0.10)
        o = odds({1: 1.4, 2: 12.0, 3: 40.0})
        d = decide_anaba_bet(r, o, s, exclude_horse_num=1)
        self.assertIsNotNone(d)
        self.assertEqual(d["horse_num"], 2)
        self.assertEqual(d["bet_kind"], "anaba")
        self.assertEqual(d["amount"], 100)

    def test_excludes_primary_horse(self):
        """primary で既に投票した馬 (exclude) は候補から外す。"""
        s = strat(anaba_min_prob=0.10)  # 馬2 (prob=0.05) を候補外にする
        r = race([0.15, 0.05])
        o = odds({1: 11.0, 2: 20.0})
        d = decide_anaba_bet(r, o, s, exclude_horse_num=1)
        self.assertIsNone(d)  # 馬2 は prob<10% で候補外、馬1 は exclude

    def test_no_qualifier_returns_none(self):
        """10 倍未満しかない、または全馬が確率閾値未満なら None。"""
        s = strat()
        r = race([0.35, 0.20, 0.10])
        o = odds({1: 2.0, 2: 3.0, 3: 8.0})  # 全て 10 倍未満
        self.assertIsNone(decide_anaba_bet(r, o, s))

    def test_max_odds_cap(self):
        """anaba_max_odds を超える超大穴は除外。"""
        s = strat(anaba_max_odds=25.0)
        r = race([0.12])
        o = odds({1: 50.0})
        self.assertIsNone(decide_anaba_bet(r, o, s))

    def test_picks_highest_ev(self):
        """複数候補がある場合は EV (prob × odds) 最大を選ぶ。"""
        s = strat()
        r = race([0.15, 0.12], horse_nums=[5, 8])
        o = odds({5: 12.0, 8: 20.0})  # ev: 1.80 vs 2.40 → 8 を選ぶ
        d = decide_anaba_bet(r, o, s)
        self.assertEqual(d["horse_num"], 8)

    def test_disabled_returns_none(self):
        s = strat(anaba_enabled=False)
        r = race([0.15])
        o = odds({1: 15.0})
        self.assertIsNone(decide_anaba_bet(r, o, s))

    def test_empty_odds_returns_none(self):
        s = strat()
        r = race([0.15])
        self.assertIsNone(decide_anaba_bet(r, pd.DataFrame(columns=["comb", "odds"]), s))


if __name__ == "__main__":
    unittest.main()
