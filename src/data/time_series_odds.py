"""時系列オッズ (JD: 単複) CSV ローダー。

JD CSV は 1 レース 1 ファイル、複数の時刻スナップショット行を含む。
カラム: レースID, 区分(snapshot index), 月日時分, 頭数, 単勝票数, 複勝票数,
       [1単, 1複Lo, 1複Hi, ..., N単, N複Lo, N複Hi]
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

_JD_META_COLS = [
    "race_id_ts", "snapshot_idx", "time_code", "horse_N", "win_votes", "show_votes",
]
MAX_HORSES = 18


def race_id_main_to_ts(main_race_id: int | str) -> int:
    """main 形式 (18 桁) → ts 形式 (16 桁) に変換。"""
    s = str(int(main_race_id))
    if len(s) >= 16:
        return int(s[:16])
    return int(s)


def derive_ts_race_ids_from_main(main_df: pd.DataFrame) -> pd.Series:
    """main DataFrame の race_id 列から ts 形式 race_id 列を作る。"""
    return main_df["race_id"].astype("int64").apply(race_id_main_to_ts).astype("int64")


def parse_jd_csv(path: str | Path, encoding: str = "shift_jis") -> pd.DataFrame:
    """単一の JD CSV を long 形式 (race × horse × snapshot) で返す。"""
    try:
        df_raw = pd.read_csv(path, encoding=encoding, header=0)
    except Exception:
        return pd.DataFrame()

    n_cols = len(df_raw.columns)
    if n_cols < 6 + 3:
        return pd.DataFrame()

    n_horses = (n_cols - 6) // 3
    new_cols = list(_JD_META_COLS)
    for n in range(1, n_horses + 1):
        new_cols.extend([f"win_odds_{n}", f"show_odds_lo_{n}", f"show_odds_hi_{n}"])
    df_raw.columns = new_cols[:n_cols]

    for col in _JD_META_COLS:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw = df_raw.dropna(subset=["race_id_ts", "snapshot_idx"]).copy()
    if df_raw.empty:
        return pd.DataFrame()
    df_raw["race_id_ts"] = df_raw["race_id_ts"].astype("int64")
    df_raw["snapshot_idx"] = df_raw["snapshot_idx"].astype(int)
    df_raw["time_code"] = pd.to_numeric(df_raw["time_code"], errors="coerce").fillna(0).astype("int64")
    df_raw["horse_N"] = pd.to_numeric(df_raw["horse_N"], errors="coerce").fillna(0).astype(int)
    df_raw["win_votes"] = pd.to_numeric(df_raw["win_votes"], errors="coerce").fillna(0).astype("int64")
    df_raw["show_votes"] = pd.to_numeric(df_raw["show_votes"], errors="coerce").fillna(0).astype("int64")

    long_frames: list[pd.DataFrame] = []
    for n in range(1, n_horses + 1):
        cols_n = [f"win_odds_{n}", f"show_odds_lo_{n}", f"show_odds_hi_{n}"]
        if not all(c in df_raw.columns for c in cols_n):
            continue
        sub = df_raw[_JD_META_COLS + cols_n].copy()
        sub.columns = list(_JD_META_COLS) + ["win_odds", "show_odds_lo", "show_odds_hi"]
        sub["horse_num"] = n
        long_frames.append(sub)

    if not long_frames:
        return pd.DataFrame()
    long_df = pd.concat(long_frames, axis=0, ignore_index=True)
    for c in ["win_odds", "show_odds_lo", "show_odds_hi"]:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce")
    long_df = long_df[long_df["win_odds"].fillna(0) > 0].reset_index(drop=True)
    return long_df


def aggregate_jd_per_horse(long_df: pd.DataFrame) -> pd.DataFrame:
    """long 形式 → 馬×レース集約特徴量。"""
    if long_df.empty:
        return pd.DataFrame()

    df = long_df.copy()
    df.sort_values(["race_id_ts", "horse_num", "snapshot_idx", "time_code"], inplace=True)
    df["show_mid"] = (df["show_odds_lo"].fillna(0) + df["show_odds_hi"].fillna(0)) / 2.0

    g = df.groupby(["race_id_ts", "horse_num"], sort=False)
    first = g.head(1).set_index(["race_id_ts", "horse_num"])
    last = g.tail(1).set_index(["race_id_ts", "horse_num"])

    stats = g.agg(
        ts_n_snapshots=("snapshot_idx", "count"),
        ts_win_min=("win_odds", "min"),
        ts_win_max=("win_odds", "max"),
        ts_win_std=("win_odds", "std"),
    )

    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce").astype(float)
        b = pd.to_numeric(b, errors="coerce").astype(float)
        return (a / b.where(b > 0)).fillna(0.0)

    feat = pd.DataFrame(index=stats.index)
    feat["ts_n_snapshots"] = stats["ts_n_snapshots"].astype("int64")
    feat["ts_win_first"] = first["win_odds"].astype(float)
    feat["ts_win_last"] = last["win_odds"].astype(float)
    feat["ts_win_min"] = stats["ts_win_min"].astype(float)
    feat["ts_win_max"] = stats["ts_win_max"].astype(float)
    feat["ts_win_std"] = stats["ts_win_std"].astype(float).fillna(0.0)
    feat["ts_win_drop_pct"] = _safe_div(feat["ts_win_first"] - feat["ts_win_last"], feat["ts_win_first"])
    feat["ts_show_lo_last"] = last["show_odds_lo"].astype(float)
    feat["ts_show_hi_last"] = last["show_odds_hi"].astype(float)
    feat["ts_show_first_mid"] = first["show_mid"].astype(float)
    feat["ts_show_last_mid"] = last["show_mid"].astype(float)
    feat["ts_show_drop_pct"] = _safe_div(feat["ts_show_first_mid"] - feat["ts_show_last_mid"], feat["ts_show_first_mid"])
    feat["ts_implied_prob_last"] = _safe_div(pd.Series(1.0, index=feat.index), feat["ts_win_last"])
    feat["ts_implied_prob_first"] = _safe_div(pd.Series(1.0, index=feat.index), feat["ts_win_first"])
    feat["ts_win_votes_last"] = pd.to_numeric(last["win_votes"], errors="coerce").fillna(0).astype("int64")
    feat["ts_show_votes_last"] = pd.to_numeric(last["show_votes"], errors="coerce").fillna(0).astype("int64")
    feat["ts_time_to_post"] = pd.to_numeric(last["time_code"], errors="coerce").fillna(0).astype("int64")

    def _late_drop(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        cut = max(1, int(len(s) * 0.75))
        early = float(s.iloc[cut - 1])
        late = float(s.iloc[-1])
        if early <= 0:
            return 0.0
        return (early - late) / early

    feat["ts_win_late_drop_pct"] = g["win_odds"].apply(_late_drop)
    feat = feat.reset_index()
    feat["_rank_first"] = feat.groupby("race_id_ts")["ts_win_first"].rank(method="min")
    feat["_rank_last"] = feat.groupby("race_id_ts")["ts_win_last"].rank(method="min")
    feat["ts_pop_rank_change"] = (feat["_rank_first"] - feat["_rank_last"]).astype(float).fillna(0.0)
    feat.drop(columns=["_rank_first", "_rank_last"], inplace=True)
    sum_imp = feat.groupby("race_id_ts")["ts_implied_prob_last"].transform("sum")
    feat["ts_implied_prob_last_norm"] = (
        pd.to_numeric(feat["ts_implied_prob_last"], errors="coerce")
        / pd.to_numeric(sum_imp, errors="coerce").where(sum_imp > 0)
    ).fillna(0.0)
    return feat


def _filename_year(filename: str) -> int | None:
    name = os.path.basename(filename)
    if not (name.startswith("JD") and name.endswith(".CSV")):
        return None
    if len(name) < 12:
        return None
    try:
        y2 = int(name[4:6])
    except ValueError:
        return None
    if 0 <= y2 < 30:
        return 2000 + y2
    if 30 <= y2 < 100:
        return 1900 + y2
    return None


def load_jd_for_years(
    ts_odds_dir: str | Path,
    years: list[int],
    cache_dir: str | Path | None = None,
    encoding: str = "shift_jis",
    verbose: bool = False,
    aggregate: bool = True,
) -> pd.DataFrame:
    """年単位で JD CSV を読み込み (キャッシュ対応)。"""
    ts_odds_dir = str(ts_odds_dir)
    if cache_dir:
        cache_dir = str(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    suffix = "agg" if aggregate else "raw"

    frames: list[pd.DataFrame] = []
    for year in sorted(set(years)):
        cache_path = (
            os.path.join(cache_dir, f"jd_{year}_{suffix}.parquet") if cache_dir else None
        )
        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f"  [cache] year {year}: loading {cache_path}")
            frames.append(pd.read_parquet(cache_path))
            continue

        y2 = f"{year % 100:02d}"
        pattern = os.path.join(ts_odds_dir, f"JD??{y2}*.CSV")
        files = sorted(glob.glob(pattern))
        files = [f for f in files if _filename_year(f) == year]
        if verbose:
            print(f"Year {year}: parsing {len(files)} files (aggregate={aggregate})")

        year_frames: list[pd.DataFrame] = []
        for fp in files:
            df = parse_jd_csv(fp, encoding=encoding)
            if df.empty:
                continue
            if aggregate:
                df = aggregate_jd_per_horse(df)
                if df.empty:
                    continue
            year_frames.append(df)

        if not year_frames:
            continue
        year_df = pd.concat(year_frames, axis=0, ignore_index=True)
        if cache_path:
            year_df.to_parquet(cache_path, index=False)
        frames.append(year_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)
