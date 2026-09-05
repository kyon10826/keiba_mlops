"""masters_client の純粋関数 (API を叩かない部分) のユニットテスト。

実行: python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.masters_client import (
    MastersVoteClient,
    extract_registered_bets,
    race_id_to_odds_id,
    race_id_to_vote_id,
)


class TestRaceIdConversion(unittest.TestCase):
    """race_id 変換 (18桁 = YYYYMMDD PP TT DD RR HH)。"""

    def test_18_digits(self):
        # 2024/07/21 場所10 回03 日08 レース12 馬06
        rid = 202407211003081206
        self.assertEqual(race_id_to_odds_id(rid), "202407211012")
        self.assertEqual(race_id_to_vote_id(rid), "202410030812")

    def test_16_digits(self):
        # HH なし 16 桁でも同じスライスで正しく変換される
        rid = 2024072110030812
        self.assertEqual(race_id_to_odds_id(rid), "202407211012")
        self.assertEqual(race_id_to_vote_id(rid), "202410030812")

    def test_string_input(self):
        self.assertEqual(race_id_to_odds_id("202407211003081206"), "202407211012")


class TestBuildBetData(unittest.TestCase):
    def test_format(self):
        bd = MastersVoteClient.build_win_bet_data("202410030812", 6, 5000)
        self.assertEqual(bd["race_id"], "202410030812")
        self.assertEqual(bd["mark"], {"6": 1})
        self.assertEqual(bd["bet"], [{"bet_id": "b1_c0_6", "money": "5000"}])

    def test_no_zero_padding_in_bet_id(self):
        # bet_id の馬番はゼロ埋めしない (公式サンプル準拠)
        bd = MastersVoteClient.build_win_bet_data("202410030812", 14, 100)
        self.assertEqual(bd["bet"][0]["bet_id"], "b1_c0_14")


class TestExtractRegisteredBets(unittest.TestCase):
    """GET /bet レスポンスの寛容パーサ。構造のバリエーションに耐えること。"""

    def test_flat_list(self):
        body = {
            "status": "OK",
            "data": [
                {"race_id": "202410030812",
                 "bet": [{"bet_id": "b1_c0_6", "money": "5000"}]},
            ],
        }
        out = extract_registered_bets(body)
        self.assertIn("202410030812", out)
        self.assertEqual(out["202410030812"][0]["bet_id"], "b1_c0_6")

    def test_nested_dict(self):
        body = {
            "status": "OK",
            "data": {
                "bet_data": [
                    {"race_id": 202410030812,
                     "bet": [{"bet_id": "b1_c0_6", "money": 5000},
                             {"bet_id": "b1_c0_7", "money": 100}]},
                ],
            },
        }
        out = extract_registered_bets(body)
        self.assertEqual(len(out["202410030812"]), 2)

    def test_empty_and_unexpected(self):
        self.assertEqual(extract_registered_bets({"status": "OK", "data": []}), {})
        self.assertEqual(extract_registered_bets("not json"), {})
        self.assertEqual(extract_registered_bets(None), {})


class TestDryRunPlaceBet(unittest.TestCase):
    def test_dry_run_does_not_call_api(self):
        client = MastersVoteClient("id", "pw", dry_run=True)
        result = client.place_win_bet("202410030812", 6, 5000)
        self.assertTrue(result.ok)
        self.assertIsNone(result.verified)
        self.assertTrue(result.raw_response.get("dry_run"))


class TestExtractRemaining(unittest.TestCase):
    def test_top_level(self):
        self.assertEqual(
            MastersVoteClient._extract_remaining({"remaining_money": "990000"}), 990000)

    def test_under_data(self):
        self.assertEqual(
            MastersVoteClient._extract_remaining(
                {"status": "OK", "data": {"remaining_money": 985000}}), 985000)

    def test_missing(self):
        self.assertIsNone(MastersVoteClient._extract_remaining({"status": "OK"}))
        self.assertIsNone(MastersVoteClient._extract_remaining(None))


if __name__ == "__main__":
    unittest.main()


class TestUnwrapDataShapes(unittest.TestCase):
    """_unwrap_data が dict/list ラップ両方から key を取り出せることを検証。"""

    def test_dict_form(self):
        from src.api.masters_client import MastersDataClient
        body = {"message":"OK", "data": {"timetable": [{"a":1}], "runtable": [{"b":2}]}}
        self.assertEqual(MastersDataClient._unwrap_data(body, "timetable"), [{"a":1}])
        self.assertEqual(MastersDataClient._unwrap_data(body, "runtable"), [{"b":2}])

    def test_list_wrapped_form(self):
        from src.api.masters_client import MastersDataClient
        body = {"message":"OK", "data": [{"timetable": [{"a":1}], "runtable": [{"b":2}]}]}
        self.assertEqual(MastersDataClient._unwrap_data(body, "timetable"), [{"a":1}])
        self.assertEqual(MastersDataClient._unwrap_data(body, "runtable"), [{"b":2}])

    def test_missing_key(self):
        from src.api.masters_client import MastersDataClient
        self.assertEqual(MastersDataClient._unwrap_data({"data": {}}, "timetable"), [])
        self.assertEqual(MastersDataClient._unwrap_data({}, "timetable"), [])
        self.assertEqual(MastersDataClient._unwrap_data(None, "timetable"), [])
