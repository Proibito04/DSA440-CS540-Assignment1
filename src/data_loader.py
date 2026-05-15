"""
Loads and merges OULAD tables into a flat feature matrix.
All feature engineering shared across modules lives here.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Target mapping: consolidate Distinction→Pass
OUTCOME_MAP = {"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1}


def load_oulad(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Returns a flat DataFrame with one row per (student, module, presentation).
    Binary target: 0 = Pass/Distinction, 1 = Fail/Withdrawn.
    """
    info = pd.read_csv(data_dir / "studentInfo.csv")
    registration = pd.read_csv(data_dir / "studentRegistration.csv")
    assessments = pd.read_csv(data_dir / "assessments.csv")
    student_assessment = pd.read_csv(data_dir / "studentAssessment.csv")
    student_vle = pd.read_csv(data_dir / "studentVle.csv")

    # --- VLE features ---
    vle_agg = (
        student_vle.groupby(["id_student", "code_module", "code_presentation"])
        .agg(total_clicks=("sum_click", "sum"), vle_days=("date", "nunique"))
        .reset_index()
    )

    # --- Assessment features ---
    assess_with_weight = student_assessment.merge(
        assessments[["id_assessment", "weight", "code_module", "code_presentation"]],
        on="id_assessment",
    )
    assess_agg = (
        assess_with_weight.groupby(["id_student", "code_module", "code_presentation"])
        .apply(
            lambda df: pd.Series(
                {
                    "weighted_score": np.average(
                        df["score"].fillna(0), weights=df["weight"]
                    ),
                    "submission_rate": df["score"].notna().mean(),
                }
            )
        )
        .reset_index()
    )

    # --- Merge everything ---
    df = (
        info.merge(vle_agg, on=["id_student", "code_module", "code_presentation"], how="left")
        .merge(assess_agg, on=["id_student", "code_module", "code_presentation"], how="left")
    )

    df["target"] = df["final_result"].map(OUTCOME_MAP)
    df = df.dropna(subset=["target"])

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (feature_cols, sensitive_cols).
    sensitive_cols are used for fairness/privacy analysis but may be dropped from training.
    """
    sensitive_cols = ["gender", "disability", "imd_band", "age_band", "highest_education"]
    feature_cols = [
        "total_clicks", "vle_days", "weighted_score", "submission_rate",
        "num_of_prev_attempts", "studied_credits",
        # Encoded categoricals added by encode_categoricals()
    ]
    return feature_cols, sensitive_cols


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encodes low-cardinality categorical columns."""
    cat_cols = ["gender", "disability", "imd_band", "age_band",
                "highest_education", "region", "code_module", "code_presentation"]
    for col in cat_cols:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype("category").cat.codes
    return df
