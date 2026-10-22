"""Tests for src.model — training and prediction."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURE_NAMES
from src.model import load_model, predict_url, train_and_evaluate


def _make_dataset(n_per_class: int = 15) -> pd.DataFrame:
    """Create a small synthetic dataset for testing.

    Legitimate URLs are short, simple hostnames.
    Phishing URLs are longer, with IPs, digits, hyphens, etc.
    """
    legit = [f"https://site{i}.com/page" for i in range(n_per_class)]
    phish = [
        f"http://192.168.0.{i}:9999/login?user=admin&token={'x'*20}"
        for i in range(n_per_class)
    ]
    urls = legit + phish
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({"url": urls, "label": labels})


class TestTrainAndEvaluate:

    def test_training_returns_metrics(self, tmp_path):
        df = _make_dataset()
        metrics = train_and_evaluate(
            df,
            model_path=tmp_path / "models",
            results_path=tmp_path / "results",
        )
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "false_positive_rate" in metrics
        assert "confusion_matrix" in metrics

    def test_model_file_created(self, tmp_path):
        df = _make_dataset()
        train_and_evaluate(
            df,
            model_path=tmp_path / "models",
            results_path=tmp_path / "results",
        )
        assert (tmp_path / "models" / "model.joblib").is_file()
        assert (tmp_path / "models" / "feature_names.json").is_file()

    def test_results_files_created(self, tmp_path):
        df = _make_dataset()
        train_and_evaluate(
            df,
            model_path=tmp_path / "models",
            results_path=tmp_path / "results",
        )
        assert (tmp_path / "results" / "metrics.json").is_file()
        assert (tmp_path / "results" / "test_predictions.csv").is_file()

    def test_too_few_samples_raises(self, tmp_path):
        df = pd.DataFrame({
            "url": ["http://a.com", "http://b.com"],
            "label": [0, 1],
        })
        with pytest.raises(ValueError, match="Too few samples"):
            train_and_evaluate(
                df,
                model_path=tmp_path / "models",
                results_path=tmp_path / "results",
            )


class TestPredictUrl:

    @pytest.fixture(autouse=True)
    def _train_model(self, tmp_path):
        """Train a tiny model before each test in this class."""
        df = _make_dataset()
        train_and_evaluate(
            df,
            model_path=tmp_path / "models",
            results_path=tmp_path / "results",
        )
        self._model_path = tmp_path / "models"

    def test_predict_returns_required_keys(self):
        clf, scaler = load_model(self._model_path)
        result = predict_url("https://example.com", clf=clf, scaler=scaler)
        assert "label" in result
        assert "label_text" in result
        assert "probability_phishing" in result
        assert "features" in result

    def test_predict_label_is_binary(self):
        clf, scaler = load_model(self._model_path)
        result = predict_url("https://example.com", clf=clf, scaler=scaler)
        assert result["label"] in (0, 1)

    def test_feature_dict_keys_match(self):
        clf, scaler = load_model(self._model_path)
        result = predict_url("https://example.com", clf=clf, scaler=scaler)
        assert set(result["features"].keys()) == set(FEATURE_NAMES)
