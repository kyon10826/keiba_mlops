"""CSVデータの読み込みおよび結合ユーティリティ。"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import yaml

from src.data.schema import COLUMN_NAMES, ERROR_CODES_EXCLUDE


def load_config(config_path: str = "config/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_year_csv(data_dir: str, year: int, encoding: str = "shift_jis",
                  file_pattern: str = "record_data_{year}.csv") -> pd.DataFrame:
    """単一年のCSVファイルを読み込む。"""
    pattern = file_pattern.format(year=year)
    # ワイルドカードパターンに対応（例: record_data_2025_*.csv）
    if "*" in pattern:
        matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not matches:
            raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
        # 複数該当する場合は最新のファイルを使用
        filepath = matches[-1]
    else:
        filepath = os.path.join(data_dir, pattern)

    df = pd.read_csv(filepath, encoding=encoding, names=COLUMN_NAMES, low_memory=False)
    return df


def load_train_data(cfg: dict) -> pd.DataFrame:
    """指定された年の学習データを読み込んで結合する。"""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"]["train_file_pattern"]

    frames = []
    for year in cfg["data"]["train_years"]:
        df = load_year_csv(data_dir, year, encoding, pattern)
        frames.append(df)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def load_valid_data(cfg: dict) -> pd.DataFrame:
    """検証用の年のデータを読み込む。"""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"]["train_file_pattern"]
    return load_year_csv(data_dir, cfg["data"]["valid_year"], encoding, pattern)


def load_test_data(cfg: dict) -> pd.DataFrame:
    """テストデータを読み込む（最新ファイル取得のためワイルドカードパターンを使用することもある）。"""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"].get("test_file_pattern", cfg["data"]["train_file_pattern"])

    frames = []
    for year in cfg["data"]["test_years"]:
        df = load_year_csv(data_dir, year, encoding, pattern)
        frames.append(df)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def split_valid_test(df: pd.DataFrame, split_month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1年分のデータを検証用（split_monthより前）とテスト用（split_month以降）に分割する。"""
    valid = df[df["month"] < split_month].reset_index(drop=True)
    test = df[df["month"] >= split_month].reset_index(drop=True)
    return valid, test


def filter_errors(df: pd.DataFrame) -> pd.DataFrame:
    """出走取消や除外を示すerror_codeを持つ行を削除する。"""
    return df[~df["error_code"].isin(ERROR_CODES_EXCLUDE)].reset_index(drop=True)


def load_all_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """設定に従って学習・検証・テストの各分割データを読み込む。

    Returns:
        (train_df, valid_df, test_df)
    """
    train = load_train_data(cfg)
    valid_full = load_valid_data(cfg)
    valid, test_from_valid = split_valid_test(valid_full, cfg["data"]["valid_split_month"])

    # test_yearsが指定されている場合は読み込んで追加する
    if cfg["data"].get("test_years"):
        test_extra = load_test_data(cfg)
        test = pd.concat([test_from_valid, test_extra], axis=0).reset_index(drop=True)
    else:
        test = test_from_valid

    return train, valid, test
