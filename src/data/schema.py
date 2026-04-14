"""Column definitions and type mappings for horse racing CSV data.

The data specification defines 47 columns per record.
"""

COLUMN_NAMES = [
    "race_id",       # レースID(新) - int - year生成用
    "year",          # 年 - int - 2桁→4桁変換
    "month",         # 月 - int
    "day",           # 日 - int
    "times",         # 回次 - int
    "place",         # 場所 - chr
    "daily",         # 日次 - chr
    "race_num",      # レース番号 - int
    "horse",         # 馬名 - chr
    "jockey_id",     # 騎手コード - int
    "horse_N",       # 頭数 - int
    "waku_num",      # 枠番 - int
    "horse_num",     # 馬番 - int
    "class_code",    # クラスコード - int
    "track_code",    # トラックコード(JV) - int
    "corner_num",    # コーナー回数 - int
    "dist",          # 距離 - int
    "state",         # 馬場状態 - chr
    "weather",       # 天候 - chr
    "age_code",      # 年齢限定(競走種別コード) - int
    "sex",           # 性別 - chr
    "age",           # 年齢 - int
    "basis_weight",  # 斤量 - num
    "blinker",       # ブリンカー - chr
    "weight",        # 馬体重 - int
    "inc_dec",       # 増減 - int
    "weight_code",   # 重量コード - int
    "win_odds",      # 単勝オッズ - num
    "rank",          # 確定着順 - int (0=未入線)
    "time_diff",     # 着差タイム - chr
    "time",          # 走破タイム(秒) - num
    "corner1_rank",  # 通過順1角 - int
    "corner2_rank",  # 通過順2角 - int
    "corner3_rank",  # 通過順3角 - int
    "corner4_rank",  # 通過順4角 - int
    "last_3F_time",  # 上がり3Fタイム - num
    "last_3F_rank",  # 上がり3F順位 - int
    "Ave_3F",        # Ave-3F - num
    "PCI",           # ペースチェンジ指数 - num
    "last_3F_time_diff",  # -3F差 - num
    "leg",           # 脚質 - chr
    "pop",           # 人気 - int
    "prize",         # 賞金 - int
    "error_code",    # 異常コード - int (0=正常,1=出走取消,2=発送除外,3=競走除外,4=競走中止,5=失格,6=落馬再騎乗,7=降着)
    "father",        # 父馬名 - chr
    "mother",        # 母馬名 - chr
    "id",            # 血統登録番号 - int (馬id 10桁)
]

DTYPE_MAP = {
    "race_id": "int64",
    "year": "int64",
    "month": "int64",
    "day": "int64",
    "times": "int64",
    "place": "str",
    "daily": "str",
    "race_num": "int64",
    "horse": "str",
    "jockey_id": "int64",
    "horse_N": "int64",
    "waku_num": "int64",
    "horse_num": "int64",
    "class_code": "int64",
    "track_code": "int64",
    "corner_num": "int64",
    "dist": "int64",
    "state": "str",
    "weather": "str",
    "age_code": "int64",
    "sex": "str",
    "age": "int64",
    "basis_weight": "float64",
    "blinker": "str",
    "weight": "float64",
    "inc_dec": "float64",
    "weight_code": "int64",
    "win_odds": "float64",
    "rank": "int64",
    "time_diff": "str",
    "time": "float64",
    "corner1_rank": "int64",
    "corner2_rank": "int64",
    "corner3_rank": "int64",
    "corner4_rank": "int64",
    "last_3F_time": "float64",
    "last_3F_rank": "int64",
    "Ave_3F": "float64",
    "PCI": "float64",
    "last_3F_time_diff": "float64",
    "leg": "str",
    "pop": "float64",
    "prize": "float64",
    "error_code": "int64",
    "father": "str",
    "mother": "str",
    "id": "int64",
}

PLACE_NAMES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

PLACE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

# 地方競馬場コード (NAR)
NAR_PLACE_CODES = {
    "門別": "30", "北見": "31", "岩見沢": "32", "帯広": "33", "旭川": "34",
    "盛岡": "35", "水沢": "36",
    "浦和": "42", "船橋": "43", "大井": "44", "川崎": "45",
    "金沢": "46", "笠松": "47", "名古屋": "48",
    "園田": "50", "姫路": "51",
    "高知": "54",
    "佐賀": "55",
}

# 全競馬場コード（JRA + NAR）
ALL_PLACE_CODES = {**PLACE_CODES, **NAR_PLACE_CODES}


def is_nar_race(race_id: str) -> bool:
    """地方競馬かどうかを判定する（場コードが11以上）"""
    rid = str(race_id).zfill(12)
    place_code = int(rid[4:6])
    return place_code >= 11

ERROR_CODES_EXCLUDE = [1, 3]  # 出走取消, 競走除外 → 返還対象
