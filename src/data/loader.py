"""CSV data loading and concatenation utilities."""

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
    """Load a single year's CSV file."""
    pattern = file_pattern.format(year=year)
    # Handle wildcard patterns (e.g. record_data_2025_*.csv)
    if "*" in pattern:
        matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not matches:
            raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
        # Use the latest file if multiple matches
        filepath = matches[-1]
    else:
        filepath = os.path.join(data_dir, pattern)

    df = pd.read_csv(filepath, encoding=encoding, names=COLUMN_NAMES, low_memory=False)
    return df


def load_train_data(cfg: dict) -> pd.DataFrame:
    """Load and concatenate training data for the specified years."""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"]["train_file_pattern"]

    frames = []
    for year in cfg["data"]["train_years"]:
        df = load_year_csv(data_dir, year, encoding, pattern)
        frames.append(df)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def load_valid_data(cfg: dict) -> pd.DataFrame:
    """Load validation year data."""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"]["train_file_pattern"]
    return load_year_csv(data_dir, cfg["data"]["valid_year"], encoding, pattern)


def load_test_data(cfg: dict) -> pd.DataFrame:
    """Load test data (may use wildcard pattern for latest file)."""
    data_dir = cfg["data"]["dir"]
    encoding = cfg["data"]["encoding"]
    pattern = cfg["data"].get("test_file_pattern", cfg["data"]["train_file_pattern"])

    frames = []
    for year in cfg["data"]["test_years"]:
        df = load_year_csv(data_dir, year, encoding, pattern)
        frames.append(df)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def split_valid_test(df: pd.DataFrame, split_month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a year's data into validation (before split_month) and test (from split_month onward)."""
    valid = df[df["month"] < split_month].reset_index(drop=True)
    test = df[df["month"] >= split_month].reset_index(drop=True)
    return valid, test


def filter_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with error_code indicating cancellation or exclusion."""
    return df[~df["error_code"].isin(ERROR_CODES_EXCLUDE)].reset_index(drop=True)


def load_all_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits according to config.

    Returns:
        (train_df, valid_df, test_df)
    """
    train = load_train_data(cfg)
    valid_full = load_valid_data(cfg)
    valid, test_from_valid = split_valid_test(valid_full, cfg["data"]["valid_split_month"])

    # If test_years are specified, load and append
    if cfg["data"].get("test_years"):
        test_extra = load_test_data(cfg)
        test = pd.concat([test_from_valid, test_extra], axis=0).reset_index(drop=True)
    else:
        test = test_from_valid

    return train, valid, test
