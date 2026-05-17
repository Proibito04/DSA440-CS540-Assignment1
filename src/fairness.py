"""
Fairness & Non-Discrimination module — Agit Sefkan Balcı
Computes fairness metrics and applies bias mitigation on OULAD predictions.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, f1_score


def compute_fairness_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    group_label: str = "group",
) -> MetricFrame:
    """
    Returns a MetricFrame with per-group accuracy, selection_rate, and f1.
    Use sensitive_features = df['disability'] or df['imd_band_enc'] etc.
    """
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "f1": lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return mf


def print_fairness_summary(mf: MetricFrame, y_true, y_pred, sensitive_features):
    """Prints overall and per-group metrics alongside disparity measures."""
    print("=== Overall Metrics ===")
    print(mf.overall)
    print("\n=== Per-Group Metrics ===")
    print(mf.by_group)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    print(f"\nDemographic Parity Difference: {dpd:.4f}  (0 = perfectly fair)")
    print(f"Equalized Odds Difference:     {eod:.4f}  (0 = perfectly fair)")


def mitigate_with_threshold_optimizer(
    model, X_train, y_train, sensitive_train,
    X_test, sensitive_test,
    constraint: str = "demographic_parity",
) -> np.ndarray:
    """
    Post-processing mitigation via ThresholdOptimizer.
    Returns mitigated predictions on X_test.
    """
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer.predict(X_test, sensitive_features=sensitive_test)
