"""Model training, evaluation and persistence for Q-Sentinel.

Provides helpers to train a scikit-learn RandomForestClassifier on
extracted URL features and to evaluate it with standard metrics.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features

RANDOM_SEED = 42
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def _featurise(urls: pd.Series) -> np.ndarray:
    """Extract feature matrix from a Series of URL strings."""
    return np.array([extract_features(u) for u in urls], dtype=np.float64)


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    model_path: Path | None = None,
    results_path: Path | None = None,
) -> dict:
    """Train a RandomForest model and return evaluation metrics.

    Parameters
    ----------
    df : DataFrame
        Must have columns ``url`` (str) and ``label`` (int 0/1).
    test_size : float
        Fraction of data reserved for testing.
    model_path : Path, optional
        Where to save the trained model + scaler.  Defaults to ``models/``.
    results_path : Path, optional
        Where to save evaluation artefacts.  Defaults to ``results/``.

    Returns
    -------
    dict
        Keys: precision, recall, f1, false_positive_rate, confusion_matrix,
        classification_report, num_train, num_test.
    """
    model_path = model_path or MODEL_DIR
    results_path = results_path or RESULTS_DIR
    model_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # --- Deduplicate -----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    dupes = before - len(df)
    if dupes:
        print(f"  Removed {dupes} duplicate URLs.")

    # --- Check class balance ---------------------------------------------
    class_counts = df["label"].value_counts()
    print(f"  Class distribution: {dict(class_counts)}")
    if len(class_counts) < 2:
        raise ValueError(
            "Training requires both phishing (1) and legitimate (0) samples."
        )

    minority_count = class_counts.min()
    if minority_count < 5:
        raise ValueError(
            f"Too few samples in the minority class ({minority_count}). "
            "Need at least 5 per class for a meaningful split."
        )

    # --- Stratified split ------------------------------------------------
    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["label"],
            random_state=RANDOM_SEED,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create a stratified split: {exc}") from exc

    print(f"  Train set: {len(train_df)}  |  Test set: {len(test_df)}")

    # --- Feature extraction ----------------------------------------------
    X_train = _featurise(train_df["url"])
    X_test = _featurise(test_df["url"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # --- Preprocessing (fit ONLY on train) --------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Train ------------------------------------------------------------
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Evaluate ---------------------------------------------------------
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["legitimate", "phishing"],
            zero_division=0,
        ),
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
    }

    # --- Persist model artefacts -----------------------------------------
    joblib.dump({"model": clf, "scaler": scaler}, model_path / "model.joblib")
    with open(model_path / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"  Model saved to {model_path / 'model.joblib'}")

    # --- Persist evaluation artefacts ------------------------------------
    with open(results_path / "metrics.json", "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f,
            indent=2,
        )
    with open(results_path / "classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])

    test_preds = test_df.copy()
    test_preds["prediction"] = y_pred
    test_preds.to_csv(results_path / "test_predictions.csv", index=False)
    print(f"  Results saved to {results_path}/")

    return metrics


def load_model(model_path: Path | None = None):
    """Load a previously trained model + scaler.

    Returns (classifier, scaler).

    Raises FileNotFoundError if the model file is absent.
    """
    model_path = model_path or MODEL_DIR
    fpath = model_path / "model.joblib"
    if not fpath.is_file():
        raise FileNotFoundError(
            f"No trained model found at {fpath}. Run train.py first."
        )
    bundle = joblib.load(fpath)
    return bundle["model"], bundle["scaler"]


def predict_url(url: str, clf=None, scaler=None) -> dict:
    """Predict whether *url* is phishing.

    Returns a dict with keys: label (int), label_text (str),
    probability (float), features (dict).
    """
    if clf is None or scaler is None:
        clf, scaler = load_model()

    features = extract_features(url)
    X = scaler.transform([features])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    feature_dict = dict(zip(FEATURE_NAMES, features))

    return {
        "label": int(pred),
        "label_text": "Potentially phishing" if pred == 1 else "Likely legitimate",
        "probability_phishing": float(proba[1]),
        "probability_legitimate": float(proba[0]),
        "features": feature_dict,
    }
