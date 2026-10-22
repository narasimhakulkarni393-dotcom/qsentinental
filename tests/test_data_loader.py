"""Tests for src.data_loader — CSV loading and validation."""

import textwrap
from pathlib import Path

import pytest

from src.data_loader import load_dataset


@pytest.fixture()
def tmp_csv(tmp_path):
    """Helper that writes a mini CSV and returns its path."""
    def _write(content: str, name: str = "test.csv") -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
        return p
    return _write


class TestLoadDataset:

    def test_loads_numeric_labels(self, tmp_csv):
        p = tmp_csv("""
            url,label
            http://good.com,0
            http://evil.com,1
            http://good2.com,0
            http://evil2.com,1
            http://good3.com,0
            http://evil3.com,1
            http://good4.com,0
            http://evil4.com,1
            http://good5.com,0
            http://evil5.com,1
        """)
        df = load_dataset(p)
        assert set(df.columns) == {"url", "label"}
        assert set(df["label"].unique()) == {0, 1}

    def test_loads_text_labels(self, tmp_csv):
        p = tmp_csv("""
            url,label
            http://good.com,legitimate
            http://evil.com,phishing
            http://good2.com,legitimate
            http://evil2.com,phishing
            http://good3.com,legitimate
            http://evil3.com,phishing
            http://good4.com,legitimate
            http://evil4.com,phishing
            http://good5.com,legitimate
            http://evil5.com,phishing
        """)
        df = load_dataset(p)
        assert set(df["label"].unique()) == {0, 1}

    def test_custom_column_names(self, tmp_csv):
        p = tmp_csv("""
            URL,Label
            http://good.com,legitimate
            http://evil.com,phishing
            http://good2.com,legitimate
            http://evil2.com,phishing
            http://good3.com,legitimate
            http://evil3.com,phishing
            http://good4.com,legitimate
            http://evil4.com,phishing
            http://good5.com,legitimate
            http://evil5.com,phishing
        """)
        df = load_dataset(p, url_col="URL", label_col="Label")
        assert len(df) == 10

    def test_missing_url_column_raises(self, tmp_csv):
        p = tmp_csv("""
            site,label
            http://a.com,0
        """)
        with pytest.raises(ValueError, match="URL column"):
            load_dataset(p)

    def test_missing_label_column_raises(self, tmp_csv):
        p = tmp_csv("""
            url,status
            http://a.com,0
        """)
        with pytest.raises(ValueError, match="Label column"):
            load_dataset(p)

    def test_unmappable_text_labels_raise(self, tmp_csv):
        p = tmp_csv("""
            url,label
            http://a.com,maybe
            http://b.com,dunno
        """)
        with pytest.raises(ValueError, match="Cannot map"):
            load_dataset(p)

    def test_single_class_raises(self, tmp_csv):
        p = tmp_csv("""
            url,label
            http://a.com,0
            http://b.com,0
        """)
        with pytest.raises(ValueError, match="both classes"):
            load_dataset(p)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.csv")
