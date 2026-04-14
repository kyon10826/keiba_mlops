"""レース条件に関する特徴量。"""

from __future__ import annotations

import pandas as pd


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース条件から導出されるレースレベルの特徴量を追加する。

    作成される特徴量:
        place_encoded: 競馬場のラベルエンコード (10 場)
        track_type: 1=芝, 2=ダート (track_code の十の位から算出)
        dist: 距離 (メートル、そのまま通す)
        dist_category: S(<1400)/M(<1800)/I(<2200)/L(<2800)/E(>=2800)
        condition_encoded: 馬場状態 (state 列) のラベルエンコード
        weather_encoded: 天候のラベルエンコード
        class_grade: class_code から求めたクラスグレード
        field_size: 出走頭数 (horse_N)
    """
    out = df.copy()

    # place_encoded (int/str が混在するデータに対して一貫して str にそろえる)
    out["place"] = out["place"].astype(str)
    places = out["place"].unique()
    place_map = {p: i for i, p in enumerate(sorted(places))}
    out["place_encoded"] = out["place"].map(place_map).astype(int)

    # track_code の十の位から track_type を算出
    out["track_type"] = (out["track_code"] // 10) % 10

    # 距離カテゴリ
    out["dist_category"] = pd.cut(
        out["dist"],
        bins=[0, 1400, 1800, 2200, 2800, 9999],
        labels=[0, 1, 2, 3, 4],  # S, M, I, L, E
        right=False,
    ).astype(int)

    # 馬場状態のエンコード (馬場状態: 良/稍重/重/不良)
    condition_map = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
    out["condition_encoded"] = out["state"].map(condition_map).fillna(0).astype(int)

    # 天候エンコード (一貫して str 型にそろえる)
    out["weather"] = out["weather"].astype(str)
    weather_vals = out["weather"].dropna().unique()
    weather_map = {w: i for i, w in enumerate(sorted(weather_vals))}
    out["weather_encoded"] = out["weather"].map(weather_map).fillna(0).astype(int)

    # クラスグレード (class_code からの簡略マッピング)
    def _class_to_grade(code):
        if code >= 100:
            return 5  # G1 クラス
        elif code >= 60:
            return 4  # G2/G3
        elif code >= 40:
            return 3  # リステッド/オープン
        elif code >= 20:
            return 2  # 条件戦
        elif code >= 10:
            return 1  # 未勝利
        return 0  # 新馬/その他
    out["class_grade"] = out["class_code"].apply(_class_to_grade)

    # 出走頭数
    out["field_size"] = out["horse_N"]

    return out, place_map, weather_map
