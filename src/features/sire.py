"""Sire (father) Bayesian statistics."""

from __future__ import annotations

import pandas as pd
from scipy.stats import beta


def compute_sire_stats(train_df: pd.DataFrame, alpha_prior: int = 2,
                       beta_prior: int = 5) -> pd.DataFrame:
    """Compute Bayesian show-rate statistics per sire.

    Also computes turf/dirt split show rates.

    Returns a DataFrame with columns:
        father, sire_show_rate, sire_lcb95,
        sire_show_rate_turf, sire_show_rate_dirt
    """
    df = train_df.copy()
    df["_show"] = ((df["rank"] >= 1) & (df["rank"] <= 3)).astype(int)
    df["_track_type"] = (df["track_code"] // 10) % 10  # 1=turf, 2=dirt

    # Overall
    overall = df.groupby("father").agg(
        place_count=("_show", "sum"),
        total_count=("_show", "count"),
    ).reset_index()

    alpha_post = overall["place_count"] + alpha_prior
    beta_post = (overall["total_count"] - overall["place_count"]) + beta_prior
    overall["sire_show_rate"] = alpha_post / (alpha_post + beta_post)
    overall["sire_lcb95"] = [
        beta.ppf(0.05, a, b) for a, b in zip(alpha_post, beta_post)
    ]

    # Turf
    turf = df[df["_track_type"] == 1].groupby("father").agg(
        turf_place=("_show", "sum"),
        turf_total=("_show", "count"),
    ).reset_index()
    turf["sire_show_rate_turf"] = (turf["turf_place"] + alpha_prior) / (
        turf["turf_total"] + alpha_prior + beta_prior
    )

    # Dirt
    dirt = df[df["_track_type"] == 2].groupby("father").agg(
        dirt_place=("_show", "sum"),
        dirt_total=("_show", "count"),
    ).reset_index()
    dirt["sire_show_rate_dirt"] = (dirt["dirt_place"] + alpha_prior) / (
        dirt["dirt_total"] + alpha_prior + beta_prior
    )

    result = overall[["father", "sire_show_rate", "sire_lcb95"]].merge(
        turf[["father", "sire_show_rate_turf"]], on="father", how="left",
    ).merge(
        dirt[["father", "sire_show_rate_dirt"]], on="father", how="left",
    )
    result["sire_show_rate_turf"] = result["sire_show_rate_turf"].fillna(
        result["sire_show_rate"]
    )
    result["sire_show_rate_dirt"] = result["sire_show_rate_dirt"].fillna(
        result["sire_show_rate"]
    )

    # Global prior row: fallback values for unknown sires
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    prior_lcb = beta.ppf(0.05, alpha_prior, beta_prior)
    global_prior = pd.DataFrame([{
        "father": "_GLOBAL_PRIOR_",
        "sire_show_rate": prior_mean,
        "sire_lcb95": prior_lcb,
        "sire_show_rate_turf": prior_mean,
        "sire_show_rate_dirt": prior_mean,
    }])
    result = pd.concat([result, global_prior], ignore_index=True)

    return result


def add_sire_features(
    df: pd.DataFrame,
    sire_stats: pd.DataFrame,
    alpha_prior: int = 2,
    beta_prior: int = 5,
) -> pd.DataFrame:
    """Merge sire stats and add father_encoded feature.

    Unknown sires (not in sire_stats) are filled with the prior mean
    instead of 0, so that cold-start horses still carry meaningful
    sire information.
    """
    out = df.copy()

    out["father"] = out["father"].fillna("").astype(str)
    fathers = out["father"].unique()
    father_map = {f: i for i, f in enumerate(sorted(fathers))}
    out["father_encoded"] = out["father"].map(father_map).fillna(0).astype(int)

    sire_stats = sire_stats.copy()
    sire_stats["father"] = sire_stats["father"].astype(str)

    # Extract prior fallback values from _GLOBAL_PRIOR_ row if available
    prior_row = sire_stats[sire_stats["father"] == "_GLOBAL_PRIOR_"]
    if not prior_row.empty:
        fill_values = {
            "sire_show_rate": prior_row.iloc[0]["sire_show_rate"],
            "sire_lcb95": prior_row.iloc[0]["sire_lcb95"],
            "sire_show_rate_turf": prior_row.iloc[0]["sire_show_rate_turf"],
            "sire_show_rate_dirt": prior_row.iloc[0]["sire_show_rate_dirt"],
        }
    else:
        # Fallback: compute prior mean directly (backward compat with old sire_stats)
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        fill_values = {col: prior_mean for col in [
            "sire_show_rate", "sire_lcb95",
            "sire_show_rate_turf", "sire_show_rate_dirt",
        ]}

    # Merge excluding _GLOBAL_PRIOR_ sentinel row
    merge_stats = sire_stats[sire_stats["father"] != "_GLOBAL_PRIOR_"]
    out = out.merge(
        merge_stats[["father", "sire_show_rate", "sire_lcb95",
                      "sire_show_rate_turf", "sire_show_rate_dirt"]],
        on="father", how="left",
    )
    for col, fill_val in fill_values.items():
        out[col] = out[col].fillna(fill_val)

    return out, father_map
