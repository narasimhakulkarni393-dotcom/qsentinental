"""Model training, evaluation, threshold calibration, and persistence for Q-Sentinel.

Provides training and evaluation pipelines for:
- LightGBM (Gradient Boosted Decision Trees)
- Random Forest
- Logistic Regression
- HistGradientBoosting

Supports:
- Random Stratified Splitting
- Domain-Disjoint Splitting (prevents domain memorization / leakage)
- Validation-based Decision Threshold Tuning to prioritize Phishing Recall (reducing False Negatives)
- Metrics: Phishing Recall, Precision, F1, F2, False Negatives, False Positives, FPR, PR-AUC, ROC-AUC, Latency
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features

# Try importing LightGBM, fallback to HistGradientBoosting if unavailable
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

class ModelBundle(tuple):
    """2-tuple (clf, scaler) for backwards-compatibility, with threshold and metadata properties."""

    def __new__(cls, clf, scaler, threshold: float = 0.5, metadata: dict | None = None):
        return super().__new__(cls, (clf, scaler))

    def __init__(self, clf, scaler, threshold: float = 0.5, metadata: dict | None = None):
        self.clf = clf
        self.scaler = scaler
        self.threshold = threshold
        self.metadata = metadata or {}


RANDOM_SEED = 42
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def _featurise(urls: pd.Series | list) -> np.ndarray:
    """Extract feature matrix from an iterable of URL strings."""
    return np.array([extract_features(u) for u in urls], dtype=np.float64)


def tune_threshold_for_recall(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    min_recall: float = 0.95,
    max_fpr: float = 0.05,
) -> float:
    """Tune classification threshold on validation set to maximize recall while controlling FPR.

    Searches thresholds in [0.10, 0.90].
    Prioritizes achieving min_recall with minimal FPR, otherwise optimizes F2 score.
    """
    thresholds = np.linspace(0.10, 0.85, 76)
    best_thresh = 0.5
    best_f2 = -1.0

    valid_candidates = []

    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)

        if rec >= min_recall and fpr <= max_fpr:
            valid_candidates.append((fpr, -rec, t))

        if f2 > best_f2 and fpr <= max_fpr * 1.5:
            best_f2 = f2
            best_thresh = float(t)

    if valid_candidates:
        # Pick threshold that meets target recall with lowest FPR
        valid_candidates.sort()
        return float(valid_candidates[0][2])

    return float(best_thresh)


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str = "lightgbm",
    split_mode: str = "domain_disjoint",
    test_size: float = 0.2,
    val_size: float = 0.1,
    target_recall: float = 0.95,
    max_fpr: float = 0.05,
    model_path: Path | None = None,
    results_path: Path | None = None,
) -> dict:
    """Train a model, tune threshold on validation data, and evaluate on held-out test data.

    Parameters
    ----------
    df : DataFrame
        Must have 'url' and 'label' columns, and optionally 'domain'.
    model_type : str
        'lightgbm', 'random_forest', 'logistic_regression', or 'hist_gradient_boosting'.
    split_mode : str
        'stratified' (random stratified) or 'domain_disjoint' (groups by domain).
    test_size : float
        Fraction of data reserved for testing.
    val_size : float
        Fraction of data reserved for threshold validation.
    model_path : Path, optional
        Directory where model artifact is saved.
    results_path : Path, optional
        Directory where evaluation artifacts are saved.
    """
    model_path = Path(model_path or MODEL_DIR)
    results_path = Path(results_path or RESULTS_DIR)
    model_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # Clean & Deduplicate
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    if "domain" not in df.columns:
        from .data_loader import _extract_domain
        df["domain"] = df["url"].apply(_extract_domain)

    class_counts = df["label"].value_counts().to_dict()
    print(f"  Class balance: {class_counts}")
    if len(df) < 10 or len(class_counts) < 2 or min(class_counts.values()) < 5:
        raise ValueError("Too few samples: Training requires at least 10 samples with both classes (minimum 5 per class).")

    # Splitting logic
    if split_mode == "domain_disjoint":
        print("  Split mode: DOMAIN-DISJOINT (prevents domain memorization)")
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
        train_val_idx, test_idx = next(gss_test.split(df, df["label"], groups=df["domain"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Validation split from train_val
        val_rel_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=RANDOM_SEED)
        train_idx, val_idx = next(gss_val.split(train_val_df, train_val_df["label"], groups=train_val_df["domain"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        print("  Split mode: RANDOM STRATIFIED")
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=RANDOM_SEED
        )
        val_rel_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_rel_size, stratify=train_val_df["label"], random_state=RANDOM_SEED
        )

    print(f"  Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature extraction (fit scaler ONLY on train)
    print("  Extracting features from URLs...")
    X_train_raw = _featurise(train_df["url"])
    X_val_raw = _featurise(val_df["url"])
    X_test_raw = _featurise(test_df["url"])

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Initialize model
    print(f"  Training model: {model_type.upper()}...")
    if model_type == "lightgbm" and _HAS_LIGHTGBM:
        clf = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_type in ("lightgbm", "hist_gradient_boosting"):
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=RANDOM_SEED,
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_SEED,
        )
    else:  # random_forest
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)

    # Threshold calibration on VALIDATION data (never touch test set)
    probs_val = clf.predict_proba(X_val)[:, 1]
    optimal_thresh = tune_threshold_for_recall(y_val, probs_val, min_recall=target_recall, max_fpr=max_fpr)
    print(f"  Calibrated Decision Threshold (Validation tuned): {optimal_thresh:.3f}")

    # Evaluate on HELD-OUT TEST data
    t0_inf = time.time()
    probs_test = clf.predict_proba(X_test)[:, 1]
    inf_latency_ms = ((time.time() - t0_inf) / len(test_df)) * 1000.0

    y_pred = (probs_test >= optimal_thresh).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs_test)) if len(np.unique(y_test)) > 1 else 1.0
    pr_auc = float(average_precision_score(y_test, probs_test)) if len(np.unique(y_test)) > 1 else 1.0

    metrics = {
        "model_type": model_type,
        "split_mode": split_mode,
        "threshold": float(optimal_thresh),
        "recall": round(rec, 4),
        "recall_phishing": round(rec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "inference_latency_ms": round(inf_latency_ms, 3),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["legitimate", "phishing"], zero_division=0
        ),
    }

    # Persist model bundle
    bundle = {
        "model": clf,
        "scaler": scaler,
        "threshold": optimal_thresh,
        "feature_names": FEATURE_NAMES,
        "model_type": model_type,
        "version": "v2.0-phiusiil-enhanced",
    }
    joblib.dump(bundle, model_path / "model.joblib")
    with open(model_path / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    # Persist metrics, test predictions and reports
    with open(results_path / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    test_preds_df = pd.DataFrame({
        "url": test_df["url"],
        "true_label": y_test,
        "predicted_label": y_pred,
        "probability_phishing": probs_test,
    })
    test_preds_df.to_csv(results_path / "test_predictions.csv", index=False)

    with open(results_path / "classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])

    print(f"  Model bundle successfully saved to {model_path / 'model.joblib'}")
    return metrics


def load_model(model_path: Path | None = None) -> ModelBundle:
    """Load the saved model bundle. Returns ModelBundle(clf, scaler) with .threshold and .metadata."""
    model_path = Path(model_path or MODEL_DIR)
    fpath = model_path / "model.joblib"
    if not fpath.is_file():
        raise FileNotFoundError(f"No trained model found at {fpath}. Run train.py first.")
    bundle = joblib.load(fpath)
    if isinstance(bundle, dict) and "model" in bundle:
        clf = bundle["model"]
        scaler = bundle["scaler"]
        threshold = bundle.get("threshold", 0.5)
        return ModelBundle(clf, scaler, threshold, bundle)
    # Legacy fallback
    return ModelBundle(bundle[0], bundle[1], 0.5, {})


def predict_url(url: str, clf=None, scaler=None, threshold: float | None = None) -> dict:
    """Predict whether a URL is phishing using the trained pipeline and calibrated threshold."""
    if clf is None or scaler is None:
        bundle = load_model()
        clf = bundle.clf
        scaler = bundle.scaler
        if threshold is None:
            threshold = bundle.threshold

    if threshold is None:
        threshold = getattr(clf, "threshold", 0.5)

    features = extract_features(url)
    X = scaler.transform([features])
    proba = clf.predict_proba(X)[0]
    p_phish = float(proba[1])

    pred = int(p_phish >= threshold)
    feature_dict = dict(zip(FEATURE_NAMES, features))

    return {
        "label": pred,
        "label_text": "Potentially Phishing" if pred == 1 else "Likely Legitimate",
        "probability_phishing": round(p_phish, 4),
        "probability_legitimate": round(float(proba[0]), 4),
        "threshold_used": round(threshold, 3),
        "features": feature_dict,
    }
