"""
Shared artifact generator for the CS540 project.

Generates base_model.pkl, X_train.pkl, X_test.pkl, and y_test.pkl into data/.
Run this once per session before using any principle notebook (03–06).

Usage
-----
From the repo root (local or Colab):
    python prepare_artifacts.py

From inside notebooks/ directory:
    %run ../prepare_artifacts.py     # Jupyter
    !python ../prepare_artifacts.py  # Colab cell

Inline (as a cell in any principle notebook):
    import prepare_artifacts
    prepare_artifacts.prepare()
"""

import sys
from pathlib import Path

# Make src/ importable regardless of where this script is called from
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from src.data_loader import load_oulad, encode_categoricals, get_feature_columns
from src.model import build_splits, train_model, evaluate_model


def prepare(data_dir: Path = None, force: bool = False, model_type: str = "rf") -> None:
    """
    Build and save the shared model artifacts.

    Parameters
    ----------
    data_dir : Path, optional
        Where to write the .pkl files. Defaults to <repo_root>/data/.
    force : bool
        If True, regenerate artifacts even if they already exist.
    model_type : "rf" | "xgb"
        Which classifier to train (see src/model.py).
    """
    if data_dir is None:
        data_dir = REPO_ROOT / "data"
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    artifacts = ["base_model.pkl", "X_train.pkl", "X_test.pkl", "y_test.pkl"]
    if not force and all((data_dir / a).exists() for a in artifacts):
        print("✔ Artifacts already present — skipping.")
        print(f"  Location: {data_dir.resolve()}")
        return

    print("Preparing shared artifacts …")

    # ── Load & engineer features ──────────────────────────────────────────
    df = load_oulad(data_dir)
    df = encode_categoricals(df)
    feature_cols, sensitive_cols = get_feature_columns(df)
    encoded_feature_cols = feature_cols + [
        c + "_enc" for c in sensitive_cols if c + "_enc" in df.columns
    ]
    print(f"  Features: {encoded_feature_cols}")

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = build_splits(df, encoded_feature_cols)
    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ── Train ─────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, model_type=model_type)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  F1-macro : {metrics['f1_macro']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    joblib.dump(model,   data_dir / "base_model.pkl")
    joblib.dump(X_train, data_dir / "X_train.pkl")
    joblib.dump(X_test,  data_dir / "X_test.pkl")
    joblib.dump(y_test,  data_dir / "y_test.pkl")

    print(f"✔ Artifacts saved → {data_dir.resolve()}")


if __name__ == "__main__":
    prepare()
