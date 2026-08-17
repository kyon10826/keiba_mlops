"""run_live の当日補完ロジック (馬体重の前走値補完) のユニットテスト。

実行: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from run_live import impute_weight_from_history


class TestImputeWeightFromHistory(unittest.TestCase):
    def setUp(self):
        # 履歴: 馬 A は 2 走 (直近 480kg)、馬 B は 1 走 (452kg)、馬 C は履歴なし
        self.hist = pd.DataFrame({
            "id": [1001, 1001, 1002],
            "race_id": [202501010101010101, 202506010101010101, 202503010101010101],
            "weight": [470.0, 480.0, 452.0],
        })
        self.runtable = pd.DataFrame({
            "place": ["札幌"] * 3, "race_num": [1] * 3,
            "horse_num": [1, 2, 3],
            "id": [1001, 1002, 1003],
            "weight": [np.nan, np.nan, np.nan],
            "inc_dec": [np.nan, np.nan, np.nan],
        })

    def test_last_known_weight(self):
        out = impute_weight_from_history(self.runtable, self.hist)
        self.assertEqual(float(out.loc[0, "weight"]), 480.0)  # 直近 (race_id 最大) の値
        self.assertEqual(float(out.loc[1, "weight"]), 452.0)
        # 履歴なし → レース内平均 (480+452)/2
        self.assertAlmostEqual(float(out.loc[2, "weight"]), 466.0)
        self.assertTrue((out["inc_dec"] == 0.0).all())

    def test_actual_weight_is_kept(self):
        """netkeiba 等で実測が取れている馬は上書きしない。"""
        rt = self.runtable.copy()
        rt.loc[0, "weight"] = 500.0
        rt.loc[0, "inc_dec"] = 4.0
        out = impute_weight_from_history(rt, self.hist)
        self.assertEqual(float(out.loc[0, "weight"]), 500.0)
        self.assertEqual(float(out.loc[0, "inc_dec"]), 4.0)

    def test_no_history_at_all_falls_back_to_default(self):
        rt = self.runtable.copy()
        rt["id"] = [9001, 9002, 9003]
        out = impute_weight_from_history(rt, self.hist)
        self.assertTrue((out["weight"] == 470.0).all())


if __name__ == "__main__":
    unittest.main()
