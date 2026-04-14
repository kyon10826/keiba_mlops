"""Jockey Bayesian statistics and rolling features."""

from __future__ import annotations

import pandas as pd
from scipy.stats import beta


def compute_jockey_stats(train_df: pd.DataFrame, alpha_prior: int = 2,
                         beta_prior: int = 5) -> pd.DataFrame:
    """Compute Bayesian show-rate statistics per jockey from training data.

    Returns a DataFrame with columns:
        jockey_id, place_count, total_count, jockey_show_rate, jockey_win_rate,
        jockey_race_count, jockey_lcb95
    """
    df = train_df.copy()
    df["_show"] = ((df["rank"] >= 1) & (df["rank"] <= 3)).astype(int)
    df["_win"] = (df["rank"] == 1).astype(int)

    stats = df.groupby("jockey_id").agg(
        place_count=("_show", "sum"),
        win_count=("_win", "sum"),
        total_count=("_show", "count"),
    ).reset_index()

    alpha_post = stats["place_count"] + alpha_prior
    beta_post = (stats["total_count"] - stats["place_count"]) + beta_prior

    stats["jockey_show_rate"] = alpha_post / (alpha_post + beta_post)
    stats["jockey_win_rate"] = stats["win_count"] / stats["total_count"].clip(lower=1)
    stats["jockey_race_count"] = stats["total_count"]
    stats["jockey_lcb95"] = [
        beta.ppf(0.05, a, b) for a, b in zip(alpha_post, beta_post)
    ]

    return stats[["jockey_id", "place_count", "total_count",
                   "jockey_show_rate", "jockey_win_rate", "jockey_race_count",
                   "jockey_lcb95"]]


def add_jockey_features(df: pd.DataFrame, jockey_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge jockey stats and add jockey_encoded feature."""
    out = df.copy()

    # Label encoding
    jockeys = out["jockey_id"].unique()
    jockey_map = {j: i for i, j in enumerate(sorted(jockeys))}
    out["jockey_encoded"] = out["jockey_id"].map(jockey_map).fillna(0).astype(int)

    # Merge Bayesian stats
    out = out.merge(
        jockey_stats[["jockey_id", "jockey_show_rate", "jockey_win_rate",
                       "jockey_race_count", "jockey_lcb95"]],
        on="jockey_id", how="left",
    )
    for col in ["jockey_show_rate", "jockey_win_rate", "jockey_race_count", "jockey_lcb95"]:
        out[col] = out[col].fillna(0)

    return out, jockey_map
