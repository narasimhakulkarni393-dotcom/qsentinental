"""Hybrid quantum-classical pipeline for Q-Sentinel.

Classical part  : lexical feature extraction, a RandomForest ranks the features,
                  and the most informative ones are scaled to rotation angles.
Quantum part    : a variational quantum circuit (one qubit per selected feature)
                  makes the phishing / legitimate decision.

The train/test split is identical to ``src.model.train_and_evaluate`` (same
dedupe, seed and stratification), so classical and quantum metrics are
directly comparable.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .features import FEATURE_NAMES, extract_features
from .model import MODEL_DIR, RANDOM_SEED, RESULTS_DIR, _featurise
from .quantum import VariationalQuantumClassifier

QUANTUM_MODEL_FILE = "quantum_model.joblib"
QUANTUM_METRICS_FILE = "quantum_metrics.json"


def _make_angle_scaler() -> object:
    """log1p -> MinMax to [0, pi] (clipped, so unseen extremes stay valid angles)."""
    return make_pipeline(
        FunctionTransformer(np.log1p),
        MinMaxScaler(feature_range=(0.0, np.pi), clip=True),
    )


def train_quantum_and_evaluate(
    df: pd.DataFrame,
    n_qubits: int = 4,
    n_layers: int = 3,
    steps: int = 120,
    batch_size: int = 64,
    lr: float = 0.1,
    max_train: int = 800,
    test_size: float = 0.2,
    model_path: Path | None = None,
    results_path: Path | None = None,
) -> dict:
    """Train the hybrid model and evaluate it on the shared test split."""
    model_path = Path(model_path or MODEL_DIR)
    results_path = Path(results_path or RESULTS_DIR)
    model_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    n_qubits = int(min(max(n_qubits, 1), len(FEATURE_NAMES)))

    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Training requires both phishing (1) and legitimate (0) samples.")
    if counts.min() < 5:
        raise ValueError(
            f"Too few samples in the minority class ({counts.min()}). "
            "Need at least 5 per class for a meaningful split."
        )

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"  Train set: {len(train_df)}  |  Test set: {len(test_df)}")

    X_train = _featurise(train_df["url"])
    X_test = _featurise(test_df["url"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # --- Classical step: pick the n most informative features -------------
    ranker = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1
    ).fit(X_train, y_train)
    feat_idx = np.sort(np.argsort(ranker.feature_importances_)[::-1][:n_qubits])
    feat_names = [FEATURE_NAMES[i] for i in feat_idx]
    print(f"  Qubit <- feature: {feat_names}")

    scaler = _make_angle_scaler().fit(X_train[:, feat_idx])
    A_train = scaler.transform(X_train[:, feat_idx])
    A_test = scaler.transform(X_test[:, feat_idx])

    # --- Quantum step: train the VQC on a stratified subsample ------------
    if len(A_train) > max_train:
        sub, _ = train_test_split(
            np.arange(len(A_train)),
            train_size=max_train,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
    else:
        sub = np.arange(len(A_train))

    vqc = VariationalQuantumClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=RANDOM_SEED,
    )
    print(f"  Training VQC on {len(sub)} samples ({n_qubits} qubits, {n_layers} layers)...")
    vqc.fit(A_train[sub], y_train[sub])

    # --- Evaluate ---------------------------------------------------------
    y_pred = vqc.predict(A_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["legitimate", "phishing"], zero_division=0
        ),
        "num_train": int(len(train_df)),
        "num_quantum_train": int(len(sub)),
        "num_test": int(len(test_df)),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "steps": steps,
        "features_used": feat_names,
        "loss_history": [float(x) for x in vqc.loss_history_],
    }

    joblib.dump(
        {"vqc": vqc, "scaler": scaler, "feature_idx": feat_idx.tolist(),
         "feature_names": feat_names},
        model_path / QUANTUM_MODEL_FILE,
    )
    with open(results_path / QUANTUM_METRICS_FILE, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
    with open(results_path / "quantum_classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])
    print(f"  Quantum model saved to {model_path / QUANTUM_MODEL_FILE}")
    return metrics


def load_quantum_model(model_path: Path | None = None) -> dict:
    """Load the saved hybrid bundle (vqc, scaler, feature indices)."""
    fpath = Path(model_path or MODEL_DIR) / QUANTUM_MODEL_FILE
    if not fpath.is_file():
        raise FileNotFoundError(
            f"No quantum model found at {fpath}. Run train_quantum.py first."
        )
    return joblib.load(fpath)


def predict_url_quantum(url: str, bundle: dict | None = None) -> dict:
    """Classify *url* with the hybrid quantum model."""
    bundle = bundle or load_quantum_model()
    feats = np.array([extract_features(url)], dtype=np.float64)
    angles = bundle["scaler"].transform(feats[:, bundle["feature_idx"]])
    p = float(bundle["vqc"].predict_proba(angles)[0])
    label = int(p >= 0.5)
    return {
        "label": label,
        "label_text": "Potentially phishing" if label == 1 else "Likely legitimate",
        "probability_phishing": p,
        "probability_legitimate": 1.0 - p,
        "features": dict(zip(bundle["feature_names"], feats[0, bundle["feature_idx"]])),
    }
