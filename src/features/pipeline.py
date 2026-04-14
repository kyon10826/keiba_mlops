"""Feature pipeline orchestration."""

from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from src.features.race import add_race_features
from src.features.horse import add_horse_features, compute_cold_start_defaults
from src.features.jockey import compute_jockey_stats, add_jockey_features
from src.features.sire import compute_sire_stats, add_sire_features
from src.features.relative import add_relative_features

# The ordered list of feature columns used for training
FEATURE_COLUMNS = [
    # Race
    "place_encoded", "track_type", "dist", "dist_category",
    "condition_encoded", "weather_encoded", "class_grade", "field_size",
    # Horse rolling
    "rank_last", "rank_rolling_3", "rank_rolling_5", "show_rate_last_5",
    "last_3f_rolling_3", "time_diff_rolling_3",
    "weight_horse", "weight_change", "race_span_days",
    "prize_cumsum", "label_momentum",
    # Jockey
    "jockey_encoded", "jockey_show_rate", "jockey_win_rate",
    "jockey_race_count", "jockey_lcb95",
    # Sire
    "father_encoded", "sire_show_rate", "sire_lcb95",
    "sire_show_rate_turf", "sire_show_rate_dirt",
    # Relative
    "odds_rank", "weight_zscore", "age_relative",
    # Raw passthrough
    "horse_num", "waku_num", "age",
]

CATEGORICAL_FEATURES = [
    "place_encoded", "track_type", "dist_category",
    "condition_encoded", "weather_encoded",
    "father_encoded", "jockey_encoded",
]


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create binary target: show (1-3 finish) = 1."""
    return ((df["rank"] >= 1) & (df["rank"] <= 3)).astype(int)


class FeaturePipeline:
    """Orchestrates feature building across all modules."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        bayesian = cfg.get("bayesian", {})
        self.alpha_prior = bayesian.get("alpha_prior", 2)
        self.beta_prior = bayesian.get("beta_prior", 5)

        # Stored during fit
        self.jockey_stats: pd.DataFrame | None = None
        self.sire_stats: pd.DataFrame | None = None
        self.cold_start_defaults: dict | None = None
        self.place_map: dict | None = None
        self.weather_map: dict | None = None
        self.jockey_map: dict | None = None
        self.father_map: dict | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Compute stats from training data."""
        self.jockey_stats = compute_jockey_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        self.sire_stats = compute_sire_stats(
            train_df, self.alpha_prior, self.beta_prior
        )
        # Compute cold-start defaults (requires rolling features)
        train_with_rolling = add_horse_features(train_df)
        self.cold_start_defaults = compute_cold_start_defaults(train_with_rolling)

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Apply all feature transformations.

        For training data, concatenate all years first, compute horse rolling features,
        then split back. For inference data, concatenate with historical data.

        Args:
            df: Input dataframe.
            is_train: If True, this is training data and stats will be recomputed.
        """
        # Race features
        out, self.place_map, self.weather_map = add_race_features(df)

        # Horse rolling features (requires sorted by id, race_id)
        out = add_horse_features(out, cold_start_defaults=self.cold_start_defaults)

        # Jockey features
        out, self.jockey_map = add_jockey_features(out, self.jockey_stats)

        # Sire features
        out, self.father_map = add_sire_features(out, self.sire_stats)

        # Relative features
        out = add_relative_features(out)

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform it."""
        self.fit(train_df)
        return self.transform(train_df, is_train=True)

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the feature columns from a transformed dataframe."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].copy()

    def save(self, path: str) -> None:
        """Persist pipeline state (stats + mappings)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "jockey_stats": self.jockey_stats,
            "sire_stats": self.sire_stats,
            "cold_start_defaults": self.cold_start_defaults,
            "place_map": self.place_map,
            "weather_map": self.weather_map,
            "jockey_map": self.jockey_map,
            "father_map": self.father_map,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, cfg: dict | None = None) -> "FeaturePipeline":
        """Load a saved pipeline."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        pipe = cls(cfg or {})
        pipe.jockey_stats = state["jockey_stats"]
        pipe.sire_stats = state["sire_stats"]
        pipe.cold_start_defaults = state.get("cold_start_defaults", None)
        pipe.place_map = state["place_map"]
        pipe.weather_map = state["weather_map"]
        pipe.jockey_map = state["jockey_map"]
        pipe.father_map = state["father_map"]
        pipe.alpha_prior = state["alpha_prior"]
        pipe.beta_prior = state["beta_prior"]
        return pipe
