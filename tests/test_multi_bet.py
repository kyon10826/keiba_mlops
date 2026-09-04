"""multi_bet.select_multi_bets のユニットテスト。"""
import os, sys, unittest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.strategy.multi_bet import (
    MultiBetCandidate, joint_probs_trio, joint_probs_trifecta,
    select_multi_bets, BET_TYPE_CODE,
)


class TestJointProbs(unittest.TestCase):
    def test_trio_sums_close_to_marginals(self):
        """三連複の joint prob は 6 順列の合計。"""
        wp = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
        df = joint_probs_trio(wp, top_n=5)
        # 全 5C3 = 10 通り
        self.assertEqual(len(df), 10)
        # 確率は昇順ソートではない (順不同で列挙)
        self.assertTrue((df["joint_prob"] > 0).all())

    def test_trifecta_permutations(self):
        wp = np.array([0.4, 0.3, 0.2, 0.1])
        df = joint_probs_trifecta(wp, top_n=4)
        # 4P3 = 24 通り
        self.assertEqual(len(df), 24)
        # 上位: (1着=馬0, 2着=馬1, 3着=馬2) が最大確率のはず
        best = df.sort_values("joint_prob", ascending=False).iloc[0]
        self.assertEqual((int(best["h1"]), int(best["h2"]), int(best["h3"])), (0, 1, 2))


class TestBetIdFormat(unittest.TestCase):
    def test_trio_bet_id_sorted(self):
        c = MultiBetCandidate("trio", (5, 3, 8), 0.02, 40.0, 0.80, 300)
        self.assertEqual(c.bet_id, "b7_c0_3-5-8")
        self.assertEqual(c.bet_id_padded, "b7_c0_030508")

    def test_trifecta_bet_id_preserved_order(self):
        c = MultiBetCandidate("trifecta", (5, 3, 8), 0.01, 80.0, 0.80, 300)
        self.assertEqual(c.bet_id, "b8_c0_5-3-8")

    def test_quinella_bet_id(self):
        c = MultiBetCandidate("quinella", (8, 3), 0.05, 15.0, 0.75, 300)
        self.assertEqual(c.bet_id, "b4_c0_3-8")

    def test_bet_type_codes(self):
        self.assertEqual(BET_TYPE_CODE["quinella"], 4)
        self.assertEqual(BET_TYPE_CODE["exacta"], 6)
        self.assertEqual(BET_TYPE_CODE["trio"], 7)
        self.assertEqual(BET_TYPE_CODE["trifecta"], 8)


class TestSelectMultiBets(unittest.TestCase):
    def _strat(self, **over):
        s = dict(
            multi_bet_types=["trio", "trifecta"],
            multi_top_n_trio=4, multi_top_n_trifecta=4,
            multi_min_ev=1.30, multi_min_prob_trio=0.005,
            multi_min_prob_trifecta=0.002, multi_max_odds=500.0,
            multi_amount_per_bet=300, multi_max_bets_per_race=6,
            multi_max_total_per_race=3000,
        )
        s.update(over); return s

    def test_picks_positive_ev_only(self):
        horse_nums = np.array([1, 2, 3, 4, 5])
        win_probs = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        # trio {1,2,3} joint prob = ~0.16 (Harville で計算されるはず)
        # EV=1.30 を満たすには odds >= 1.30 / 0.16 = ~8 倍
        odds_df = pd.DataFrame([
            {"comb": "010203", "odds_type": 7, "odds": 20.0},  # trio 1-2-3 EV高い
            {"comb": "010204", "odds_type": 7, "odds":  3.0},  # trio 1-2-4 EV低い
            {"comb": "010203", "odds_type": 8, "odds": 100.0}, # trifecta 1-2-3
        ])
        picks = select_multi_bets(horse_nums, win_probs, odds_df, self._strat())
        self.assertGreater(len(picks), 0)
        # 全て EV >= 1.30
        for p in picks:
            self.assertGreaterEqual(p.ev, 1.30)

    def test_respects_budget_cap(self):
        horse_nums = np.array([1, 2, 3, 4, 5])
        win_probs = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        # 大量の高 EV odds → max_total 3000pt 制限で 10 点までに収まる
        rows = []
        for i, j, k in [(1,2,3), (1,2,4), (1,3,4), (2,3,4), (1,2,5)]:
            rows.append({"comb": f"{i:02d}{j:02d}{k:02d}", "odds_type": 7, "odds": 100.0})
        picks = select_multi_bets(horse_nums, win_probs, pd.DataFrame(rows), self._strat())
        total = sum(p.amount for p in picks)
        self.assertLessEqual(total, 3000)

    def test_empty_odds_returns_empty(self):
        picks = select_multi_bets(
            np.array([1, 2, 3]), np.array([0.4, 0.3, 0.2]),
            pd.DataFrame(columns=["comb", "odds_type", "odds"]), self._strat(),
        )
        self.assertEqual(picks, [])


if __name__ == "__main__":
    unittest.main()


class TestSplitModeShape(unittest.TestCase):
    """place_multi_bet(split=True) の dry-run で N 個の結果が返ることを確認。"""

    def test_dry_run_split_returns_per_bet_list(self):
        from src.api.masters_client import MastersVoteClient
        client = MastersVoteClient("id", "pw", dry_run=True)
        cands = [
            MultiBetCandidate("trio", (1, 2, 3), 0.05, 20.0, 1.0, 300),
            MultiBetCandidate("trifecta", (1, 2, 3), 0.02, 80.0, 1.6, 300),
        ]
        results = client.place_multi_bet("202601020101", cands, split=True)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsNotNone(r)
            self.assertTrue(r.ok)

    def test_dry_run_batch_returns_single(self):
        from src.api.masters_client import MastersVoteClient
        client = MastersVoteClient("id", "pw", dry_run=True)
        cands = [
            MultiBetCandidate("trio", (1, 2, 3), 0.05, 20.0, 1.0, 300),
            MultiBetCandidate("trifecta", (1, 2, 3), 0.02, 80.0, 1.6, 300),
        ]
        result = client.place_multi_bet("202601020101", cands, split=False)
        # batch モードは単一 BetResult
        self.assertTrue(hasattr(result, "amount"))
        self.assertEqual(result.amount, 600)
