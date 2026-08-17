"""normalize_runtable (当日出馬表 API → record_data スキーマ) のユニットテスト。

実 API のレスポンス (2026-08-09 に確認した 14 列) を模した入力で検証する。
実行: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.api.masters_client import RUNTABLE_API_COLUMNS, normalize_runtable
from src.data.schema import COLUMN_NAMES


def make_api_tables():
    """実 API と同じ形の timetable / runtable (2 レース) を作る。"""
    timetable = pd.DataFrame([
        {"year": 2026, "month": 8, "day": 9, "place": "札幌", "race_num": 1,
         "start_time": "10:00", "class_code": 7, "track_code": 24},
        {"year": 2026, "month": 8, "day": 9, "place": "新潟", "race_num": 3,
         "start_time": "10:40", "class_code": 23, "track_code": 17},
    ])
    rows = [
        # place, race_num, horse_num, dist, horse, sex, age, jockey, loaf_weight,
        # father, mother, id(10桁化済), waku_num, race_id(18桁)
        ("札幌", 1, 1, 1700, "馬A", "牡", 2, "吉田隼人", 55.0, "父X", "母X",
         2024104861, 1, 202608090101060101),
        ("札幌", 1, 2, 1700, "馬B", "牝", 2, "岩田康誠", 54.0, "父Y", "母Y",
         2024106587, 2, 202608090101060102),
        ("新潟", 3, 1, 1200, "馬C", "牡", 3, "戸崎圭太", 57.0, "父Z", "母Z",
         2023101111, 1, 202608090402070301),
    ]
    runtable = pd.DataFrame(rows, columns=RUNTABLE_API_COLUMNS)
    return timetable, runtable


class TestNormalizeRuntable(unittest.TestCase):
    def setUp(self):
        self.timetable, self.runtable = make_api_tables()
        self.out = normalize_runtable(self.runtable, self.timetable)

    def test_all_record_columns_present(self):
        missing = [c for c in COLUMN_NAMES if c not in self.out.columns]
        self.assertEqual(missing, [])
        self.assertEqual(len(self.out), 3)  # 行数は不変

    def test_timetable_merge(self):
        """class_code / track_code はレース単位で timetable から結合される。"""
        sapporo = self.out[self.out["place"] == "札幌"]
        niigata = self.out[self.out["place"] == "新潟"]
        self.assertTrue((sapporo["class_code"] == 7).all())
        self.assertTrue((sapporo["track_code"] == 24).all())
        self.assertTrue((niigata["class_code"] == 23).all())
        self.assertTrue((niigata["track_code"] == 17).all())

    def test_date_fields(self):
        """year は record_data 慣習の 2 桁、month/day/times/daily は race_id 由来。"""
        row = self.out.iloc[0]
        self.assertEqual(int(row["year"]), 26)
        self.assertEqual(int(row["month"]), 8)
        self.assertEqual(int(row["day"]), 9)
        self.assertEqual(int(row["times"]), 1)   # race_id[10:12]
        self.assertEqual(int(row["daily"]), 6)   # race_id[12:14]

    def test_derived_columns(self):
        self.assertEqual(float(self.out.iloc[0]["basis_weight"]), 55.0)
        sapporo = self.out[self.out["place"] == "札幌"]
        self.assertTrue((sapporo["horse_N"] == 2).all())
        self.assertTrue((self.out["jockey_id"] == 0).all())  # netkeiba 補完前は 0
        self.assertTrue((self.out["state"] == "良").all())
        self.assertTrue((self.out["weather"] == "晴").all())
        self.assertTrue(self.out["weight"].isna().all())     # 朝は未公開
        self.assertTrue((self.out["rank"] == 0).all())

    def test_idempotent_on_record_schema(self):
        """既に record_data スキーマなら (リプレイ) 無変更で返す。"""
        again = normalize_runtable(self.out, self.timetable)
        pd.testing.assert_frame_equal(again, self.out)


if __name__ == "__main__":
    unittest.main()
