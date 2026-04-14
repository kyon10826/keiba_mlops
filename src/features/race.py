"""Race-condition features."""

from __future__ import annotations

import pandas as pd


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race-level features derived from race conditions.

    Features created:
        place_encoded: Label-encoded racecourse (10 venues)
        track_type: 1=turf, 2=dirt (from track_code tens digit)
        dist: Distance in meters (passthrough)
        dist_category: S(<1400)/M(<1800)/I(<2200)/L(<2800)/E(>=2800)
        condition_encoded: Track condition (state column) label-encoded
        weather_encoded: Weather label-encoded
        class_grade: Grade from class_code
        field_size: Number of runners (horse_N)
    """
    out = df.copy()

    # place_encoded (ensure consistent str type for mixed int/str data)
    out["place"] = out["place"].astype(str)
    places = out["place"].unique()
    place_map = {p: i for i, p in enumerate(sorted(places))}
    out["place_encoded"] = out["place"].map(place_map).astype(int)

    # track_type from track_code tens digit
    out["track_type"] = (out["track_code"] // 10) % 10

    # dist_category
    out["dist_category"] = pd.cut(
        out["dist"],
        bins=[0, 1400, 1800, 2200, 2800, 9999],
        labels=[0, 1, 2, 3, 4],  # S, M, I, L, E
        right=False,
    ).astype(int)

    # condition_encoded (馬場状態: 良/稍重/重/不良)
    condition_map = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
    out["condition_encoded"] = out["state"].map(condition_map).fillna(0).astype(int)

    # weather_encoded (ensure consistent str type)
    out["weather"] = out["weather"].astype(str)
    weather_vals = out["weather"].dropna().unique()
    weather_map = {w: i for i, w in enumerate(sorted(weather_vals))}
    out["weather_encoded"] = out["weather"].map(weather_map).fillna(0).astype(int)

    # class_grade (simplified mapping from class_code)
    def _class_to_grade(code):
        if code >= 100:
            return 5  # G1-level
        elif code >= 60:
            return 4  # G2/G3
        elif code >= 40:
            return 3  # Listed/OP
        elif code >= 20:
            return 2  # Conditions
        elif code >= 10:
            return 1  # Maiden
        return 0  # New/Other
    out["class_grade"] = out["class_code"].apply(_class_to_grade)

    # field_size
    out["field_size"] = out["horse_N"]

    return out, place_map, weather_map
