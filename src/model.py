"""
Base prediction model: trains and evaluates a Random Forest classifier on OULAD features.
Returns a fitted model and train/test splits for use by all principle modules.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Returns X_train, X_test, y_train, y_test."""
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Trains a Random Forest and returns the fitted model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Returns a dict of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, target_names=["Pass", "At-Risk"]),
    }
