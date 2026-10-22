"""Tests for src.features — URL feature extraction."""

import pytest
from src.features import FEATURE_NAMES, extract_features


class TestExtractFeatures:
    """Unit tests for extract_features()."""

    def test_vector_length_matches_feature_names(self):
        """Feature vector length must always equal len(FEATURE_NAMES)."""
        vec = extract_features("https://example.com")
        assert len(vec) == len(FEATURE_NAMES)

    def test_empty_string_returns_zeros(self):
        vec = extract_features("")
        assert vec == [0.0] * len(FEATURE_NAMES)

    def test_none_input_returns_zeros(self):
        vec = extract_features(None)
        assert vec == [0.0] * len(FEATURE_NAMES)

    def test_whitespace_only_returns_zeros(self):
        vec = extract_features("   ")
        assert vec == [0.0] * len(FEATURE_NAMES)

    def test_url_with_scheme(self):
        vec = extract_features("https://www.example.com/path?q=1")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["is_https"] == 1.0
        assert feat["num_query_params"] >= 1.0
        assert feat["url_length"] > 0

    def test_url_without_scheme(self):
        """URLs missing a scheme should still be parsed."""
        vec = extract_features("www.example.com/page")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["hostname_length"] > 0
        # Without scheme the parser prepends http://, so is_https == 0.
        assert feat["is_https"] == 0.0

    def test_ip_address_detected_v4(self):
        vec = extract_features("http://192.168.1.1/admin")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["is_ip_address"] == 1.0

    def test_regular_hostname_not_ip(self):
        vec = extract_features("https://google.com")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["is_ip_address"] == 0.0

    def test_unusual_port(self):
        vec = extract_features("http://example.com:9999/path")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["has_unusual_port"] == 1.0

    def test_standard_port(self):
        vec = extract_features("http://example.com:80/path")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["has_unusual_port"] == 0.0

    def test_at_symbol_counted(self):
        vec = extract_features("http://user@evil.com")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["num_at_symbols"] >= 1.0

    def test_hyphens_counted(self):
        vec = extract_features("http://my-phishing-site.com")
        feat = dict(zip(FEATURE_NAMES, vec))
        assert feat["num_hyphens"] >= 2.0

    def test_subdomains_counted(self):
        vec = extract_features("http://a.b.c.example.com")
        feat = dict(zip(FEATURE_NAMES, vec))
        # a.b.c.example.com -> 3 subdomains
        assert feat["num_subdomains"] == 3.0

    def test_consistent_dimensions_across_inputs(self):
        """All inputs must produce the same vector length."""
        urls = [
            "", "x", "http://a.com", "https://b.c.d.e.f.g.com:12345/p?q=1&r=2",
            "not-a-url-at-all", "http://192.168.0.1",
        ]
        lengths = {len(extract_features(u)) for u in urls}
        assert lengths == {len(FEATURE_NAMES)}
