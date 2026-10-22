"""Tests for src.quantum and src.quantum_model."""

import numpy as np
import pandas as pd
import pytest

from src.quantum import VariationalQuantumClassifier
from src.quantum_model import (
    load_quantum_model,
    predict_url_quantum,
    train_quantum_and_evaluate,
)


class TestSimulator:

    def test_state_norm_preserved(self):
        vqc = VariationalQuantumClassifier(n_qubits=3, n_layers=2)
        X = np.random.default_rng(0).uniform(0, np.pi, size=(5, 3))
        state = vqc._encode(X)
        params = np.random.default_rng(1).uniform(-np.pi, np.pi, size=(2, 3, 2))
        for layer in range(2):
            for q in range(3):
                from src.quantum import _apply_gate, _ry, _rz
                state = _apply_gate(state, _ry(params[layer, q, 0]), q, 3)
                state = _apply_gate(state, _rz(params[layer, q, 1]), q, 3)
            for perm in vqc._perms:
                state = state[:, perm]
        norms = (np.abs(state) ** 2).sum(axis=1)
        assert np.allclose(norms, 1.0)

    def test_ry_pi_flips_qubit0(self):
        vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1)
        enc = vqc._encode(np.array([[np.pi, 0.0]]))
        f = vqc._expectation(enc, np.zeros((1, 2, 2)))
        # |1> on qubit 0 -> <Z> = -1; CNOT leaves qubit 0 untouched.
        assert f[0] == pytest.approx(-1.0, abs=1e-9)

    def test_parameter_shift_matches_finite_difference(self):
        vqc = VariationalQuantumClassifier(n_qubits=3, n_layers=2)
        rng = np.random.default_rng(3)
        X = rng.uniform(0, np.pi, size=(8, 3))
        y = rng.integers(0, 2, size=8).astype(float)
        params = rng.uniform(-np.pi, np.pi, size=(2, 3, 2))
        enc = vqc._encode(X)
        _, grad = vqc._loss_and_grad(enc, y, params)

        eps = 1e-5
        for idx in [(0, 0, 0), (1, 2, 1), (0, 1, 1)]:
            hi, lo = params.copy(), params.copy()
            hi[idx] += eps
            lo[idx] -= eps
            num = (vqc._loss_and_grad(enc, y, hi)[0] - vqc._loss_and_grad(enc, y, lo)[0]) / (2 * eps)
            assert grad[idx] == pytest.approx(num, abs=1e-5)


class TestClassifier:

    def test_learns_simple_threshold(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, np.pi, size=(200, 2))
        y = (X[:, 0] > np.pi / 2).astype(int)
        vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=2, steps=60, seed=1)
        vqc.fit(X, y)
        acc = (vqc.predict(X) == y).mean()
        assert acc > 0.8
        assert vqc.loss_history_[-1] < vqc.loss_history_[0]

    def test_wrong_feature_count_raises(self):
        vqc = VariationalQuantumClassifier(n_qubits=3)
        with pytest.raises(ValueError):
            vqc.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            VariationalQuantumClassifier().predict_proba(np.zeros((1, 4)))


def _make_dataset(n: int = 20) -> pd.DataFrame:
    legit = [f"https://site{i}.com/page" for i in range(n)]
    phish = [f"http://192.168.0.{i}:9999/login?user=admin&token={'x' * 20}" for i in range(n)]
    return pd.DataFrame({"url": legit + phish, "label": [0] * n + [1] * n})


class TestHybridPipeline:

    def test_train_save_and_predict(self, tmp_path):
        metrics = train_quantum_and_evaluate(
            _make_dataset(),
            n_qubits=3, n_layers=2, steps=25,
            model_path=tmp_path / "m", results_path=tmp_path / "r",
        )
        assert metrics["n_qubits"] == 3
        assert len(metrics["features_used"]) == 3
        assert (tmp_path / "m" / "quantum_model.joblib").is_file()
        assert (tmp_path / "r" / "quantum_metrics.json").is_file()

        bundle = load_quantum_model(tmp_path / "m")
        res = predict_url_quantum("https://example.com", bundle)
        assert res["label"] in (0, 1)
        assert 0.0 <= res["probability_phishing"] <= 1.0
        assert set(res["features"]) == set(metrics["features_used"])

    def test_too_few_samples_raises(self, tmp_path):
        df = pd.DataFrame({"url": ["http://a.com", "http://b.com"], "label": [0, 1]})
        with pytest.raises(ValueError, match="Too few samples"):
            train_quantum_and_evaluate(df, model_path=tmp_path, results_path=tmp_path)

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_quantum_model(tmp_path)
