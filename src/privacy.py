"""
Privacy module — Edoardo Balzano
Privacy-aware preprocessing: sensitive feature handling and explanation-time privacy guards.
"""

import pandas as pd
import numpy as np


SENSITIVE_COLS = ["gender", "disability", "imd_band", "age_band", "highest_education", "region"]


def apply_feature_masking(df: pd.DataFrame, drop_sensitive: bool = True) -> pd.DataFrame:
    """
    Removes or generalizes sensitive demographic columns before model training.
    When drop_sensitive=True, columns are removed entirely (maximum privacy).
    When False, columns are binned/generalized (utility-privacy tradeoff).
    """
    df = df.copy()
    if drop_sensitive:
        cols_to_drop = [c for c in SENSITIVE_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
    else:
        # Generalize imd_band into 3 buckets instead of 10
        if "imd_band" in df.columns:
            mapping = {
                "0-10%": "Low", "10-20%": "Low", "20-30%": "Low",
                "30-40%": "Medium", "40-50%": "Medium", "50-60%": "Medium",
                "60-70%": "High", "70-80%": "High", "80-90%": "High", "90-100%": "High",
            }
            df["imd_band_generalized"] = df["imd_band"].map(mapping).fillna("Unknown")
            df = df.drop(columns=["imd_band"])
    return df


def suppress_sensitive_from_explanation(
    shap_values: pd.Series,
    sensitive_encoded_cols: list[str],
) -> pd.Series:
    """
    Removes sensitive encoded columns from a SHAP Series before surfacing to users.
    Used by explainability module to enforce privacy-aware explanations.
    """
    return shap_values.drop(
        labels=[c for c in sensitive_encoded_cols if c in shap_values.index],
        errors="ignore",
    )


def check_k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str], k: int = 5) -> pd.DataFrame:
    """
    Returns groups that violate k-anonymity (fewer than k records with the same QI combination).
    These rows represent re-identification risk.
    """
    present_qi = [q for q in quasi_identifiers if q in df.columns]
    counts = df.groupby(present_qi).size().reset_index(name="count")
    violations = counts[counts["count"] < k]
    if violations.empty:
        print(f"k-anonymity satisfied for k={k} across {present_qi}")
    else:
        print(f"{len(violations)} quasi-identifier groups violate k={k}:")
        print(violations)
    return violations
