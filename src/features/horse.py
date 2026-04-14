"""馬単位のローリング成績特徴量。

データリークを防ぐため、すべてのローリング集計で shift(1) を使用している。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# コールドスタート補完の対象となるローリング特徴量。
# 過去レースが無い馬ではこれらが 0 か NaN となるため、
# 種牡馬平均で埋めることでモデルに妥当な事前分布を与える。
COLD_START_FILL_FEATURES = [
    "rank_last",
    "rank_rolling_3",
    "rank_rolling_5",
    "show_rate_last_5",
    "last_3f_rolling_3",
    "time_diff_rolling_3",
]

# 初出走馬では 0 のままにしておく特徴量(0 が意味的に正しい)。
# race_span_days=0  → 過去レースなし
# prize_cumsum=0    → 獲得賞金なし
# label_momentum=0  → トレンドデータなし
_COLD_START_SKIP = {"race_span_days", "prize_cumsum", "label_momentum"}


def compute_cold_start_defaults(
    train_df: pd.DataFrame,
    sire_stats_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """学習データから種牡馬別のローリング特徴量平均を算出する。

    *実際に* 過去レース履歴を持つ行(``rank_last > 0`` の行)のみを使用し、
    初出走時の埋め合わせ値は除外する。

    Args:
        train_df: ``add_horse_features()`` 適用 **後** の学習用 DataFrame
            (ローリング列を含んでいる必要がある)。
        sire_stats_df: 未使用。将来拡張のため予約。

    Returns:
        ``{father_name: {feature: mean_value, ...}, ...}``
        特殊キー ``"_GLOBAL_"`` には、種牡馬が未知の場合に使うフォールバック
        用の全体平均が格納される。
    """
    if "rank_last" not in train_df.columns:
        return {"_GLOBAL_": {f: 0.0 for f in COLD_START_FILL_FEATURES}}

    # 実際に過去レースデータを持つ行のみを残す。
    valid = train_df[train_df["rank_last"] > 0].copy()
    if valid.empty:
        return {"_GLOBAL_": {f: 0.0 for f in COLD_START_FILL_FEATURES}}

    valid["father"] = valid["father"].astype(str)

    # グローバルフォールバック
    global_avgs: dict[str, float] = {}
    for feat in COLD_START_FILL_FEATURES:
        if feat in valid.columns:
            global_avgs[feat] = float(valid[feat].mean())
        else:
            global_avgs[feat] = 0.0

    # 種牡馬ごとの平均
    defaults: dict[str, dict[str, float]] = {"_GLOBAL_": global_avgs}
    for father, grp in valid.groupby("father"):
        avgs: dict[str, float] = {}
        for feat in COLD_START_FILL_FEATURES:
            if feat in grp.columns:
                avgs[feat] = float(grp[feat].mean())
            else:
                avgs[feat] = 0.0
        defaults[str(father)] = avgs

    return defaults


def apply_cold_start_defaults(
    df: pd.DataFrame,
    defaults: dict[str, dict[str, float]],
    father_col: str = "father",
) -> pd.DataFrame:
    """種牡馬ベースのデフォルト値でコールドスタート時のローリング特徴量を埋める。

    ``rank_last`` が 0 または NaN の行は「コールドスタート」とみなされる
    (その馬がデータセット内で過去レース記録を持たないことを意味する)。

    そのような各行について、*defaults* から馬の種牡馬を検索し、
    0/NaN の特徴量値を種牡馬平均で置き換える。
    種牡馬が見つからない場合は ``"_GLOBAL_"`` フォールバックを使用する。

    Args:
        df: ローリング特徴量が算出済みの DataFrame。
        defaults: :func:`compute_cold_start_defaults` の出力。
        father_col: 種牡馬の列名。

    Returns:
        コールドスタート特徴量を埋めた DataFrame。
    """
    out = df.copy()

    cold_mask = (out["rank_last"] == 0) | (out["rank_last"].isna())
    if not cold_mask.any():
        return out

    out[father_col] = out[father_col].astype(str)
    global_defaults = defaults.get("_GLOBAL_", {})
    cold_idx = out[cold_mask].index
    fathers = out.loc[cold_idx, father_col]

    for feat in COLD_START_FILL_FEATURES:
        if feat not in out.columns:
            continue
        # 種牡馬ルックアップから行ごとのデフォルト値を構築する。
        fill_values = fathers.map(
            lambda f, _feat=feat: defaults.get(f, global_defaults).get(_feat, 0.0)
        )
        # 現在値が 0 または NaN の位置にのみ上書きする。
        needs_fill = cold_mask & ((out[feat] == 0) | (out[feat].isna()))
        out.loc[needs_fill, feat] = fill_values.reindex(out.loc[needs_fill].index)

    return out


def add_horse_features(
    df: pd.DataFrame,
    cold_start_defaults: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """馬ごとのローリング成績特徴量を追加する。

    呼び出し前に DataFrame は (id, race_id) でソートされている必要がある。

    Args:
        df: 入力 DataFrame。
        cold_start_defaults: 指定された場合、コールドスタート馬 (rank_last==0)
            のローリング特徴量をこの辞書 (:func:`compute_cold_start_defaults`
            の出力) に基づく種牡馬平均で埋める。
            ``None`` (デフォルト) の場合は元の実装と同じ挙動となる。

    作成される特徴量:
        rank_last: 前走の着順
        rank_rolling_3: 直近 3 走の平均着順
        rank_rolling_5: 直近 5 走の平均着順
        show_rate_last_5: 直近 5 走の複勝率 (1〜3 着)
        last_3f_rolling_3: 直近 3 走の上がり 3F 平均
        time_diff_rolling_3: 直近 3 走のタイム差平均
        weight_horse: 馬体重 (そのまま通す。名前のみ明確化のため変更)
        weight_change: 馬体重増減 (inc_dec)
        race_span_days: 前走からの経過日数
        prize_cumsum: 獲得賞金の累積
        label_momentum: 直近 2 走における着順の変化
    """
    out = df.sort_values(["id", "race_id"]).copy()
    g = out.groupby("id")

    # シフト済み着順 (リーク回避)
    out["rank_last"] = g["rank"].shift(1).fillna(0)

    # 平均着順のローリング (馬をまたいだ汚染を防ぐためグループ化)
    shifted_rank = g["rank"].shift(1)
    out["rank_rolling_3"] = shifted_rank.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["rank_rolling_5"] = shifted_rank.groupby(out["id"]).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # レースごとの複勝フラグ (1〜3 着)
    out["_show_flag"] = ((out["rank"] >= 1) & (out["rank"] <= 3)).astype(float)
    shifted_show = g["_show_flag"].shift(1)
    out["show_rate_last_5"] = shifted_show.groupby(out["id"]).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # 上がり 3F のローリング
    shifted_3f = g["last_3F_time"].shift(1)
    out["last_3f_rolling_3"] = shifted_3f.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # タイム差のローリング
    out["_time_diff_num"] = pd.to_numeric(out["time_diff"], errors="coerce")
    shifted_td = g["_time_diff_num"].shift(1)
    out["time_diff_rolling_3"] = shifted_td.groupby(out["id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # 馬体重をそのまま通す
    out["weight_horse"] = out["weight"].astype(float)

    # 馬体重増減
    out["weight_change"] = out["inc_dec"].astype(float).fillna(0)

    # 前走からの経過日数 (概算: year/month/day を使用)
    out["_race_date"] = pd.to_datetime(
        out["year"].astype(str).apply(lambda y: ("20" + y) if len(y) <= 2 else y)
        + "-" + out["month"].astype(str).str.zfill(2)
        + "-" + out["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    out["race_span_days"] = g["_race_date"].diff().dt.days.fillna(0)

    # 累積賞金 (リーク防止のためシフト)
    out["prize_cumsum"] = g["prize"].apply(lambda s: s.shift(1).cumsum().fillna(0)).values

    # ラベルモメンタム: 直近 2 走における着順変化
    rank_shift1 = g["rank"].shift(1)
    rank_shift2 = g["rank"].shift(2)
    out["label_momentum"] = (rank_shift1 - rank_shift2).fillna(0)

    # 一時列のクリーンアップ
    out.drop(columns=["_show_flag", "_time_diff_num", "_race_date"], inplace=True)

    # 指定されていればコールドスタートのデフォルト値を適用する (学習時は no-op)。
    if cold_start_defaults is not None:
        out = apply_cold_start_defaults(out, cold_start_defaults)

    return out
