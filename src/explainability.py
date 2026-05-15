"""
Transparency & Explainability module — Çağrı Altınüzengi
Implements SHAP (global + local) and LIME explanations with dual-view output.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Sensitive features that must be suppressed from student-facing explanations
_SENSITIVE_FEATURES = {"imd_band_enc", "age_band_enc", "gender_enc", "disability_enc"}


def build_shap_explainer(model, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """Returns a SHAP TreeExplainer fitted on the training set."""
    return shap.TreeExplainer(model, X_train)


def plot_global_importance(explainer: shap.TreeExplainer, X_test: pd.DataFrame, max_display: int = 15):
    """Beeswarm summary plot — shows global feature importance across the test set."""
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values[:, :, 1], X_test, max_display=max_display, show=True)


def explain_instance_instructor(
    explainer: shap.TreeExplainer,
    instance: pd.Series,
    feature_names: list[str],
) -> None:
    """
    Instructor view: full SHAP waterfall plot with all feature contributions.
    Intended for educators who need actionable, detailed information.
    """
    shap_values = explainer(instance.to_frame().T)
    print("=== INSTRUCTOR VIEW ===")
    print(f"Risk Score: {shap_values.values[0, :, 1].sum():.3f} (base = {shap_values.base_values[0, 1]:.3f})")
    shap.waterfall_plot(shap_values[0, :, 1], show=True)


def explain_instance_student(
    explainer: shap.TreeExplainer,
    instance: pd.Series,
    feature_names: list[str],
    top_n: int = 3,
) -> str:
    """
    Student view: plain-language explanation with top-N non-sensitive factors.
    Sensitive features (socioeconomic, demographic) are suppressed to protect privacy.
    Returns a human-readable string.
    """
    shap_values = explainer(instance.to_frame().T)
    contributions = pd.Series(
        shap_values.values[0, :, 1], index=feature_names
    )
    # Remove sensitive features from student-facing output
    visible = contributions.drop(
        labels=[f for f in _SENSITIVE_FEATURES if f in contributions.index],
        errors="ignore",
    )
    top = visible.abs().nlargest(top_n).index
    lines = ["=== YOUR RISK SUMMARY ==="]
    for feat in top:
        direction = "increasing" if contributions[feat] > 0 else "decreasing"
        lines.append(f"  - {feat}: {direction} your risk")
    lines.append("\nNote: This is a decision-support tool. Speak to your advisor for guidance.")
    return "\n".join(lines)


def build_lime_explainer(X_train: pd.DataFrame, feature_names: list[str]) -> LimeTabularExplainer:
    """Returns a LIME tabular explainer trained on the training distribution."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["Pass", "At-Risk"],
        mode="classification",
        random_state=42,
    )


def explain_lime_instance(
    lime_explainer: LimeTabularExplainer,
    model,
    instance: pd.Series,
    num_features: int = 8,
):
    """Returns a LIME explanation object for one instance."""
    exp = lime_explainer.explain_instance(
        instance.values,
        model.predict_proba,
        num_features=num_features,
        labels=[1],
    )
    exp.show_in_notebook(show_table=True)
    return exp
