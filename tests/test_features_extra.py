"""Regression tests for feature-extraction edge cases."""

from src.features import FEATURE_NAMES, extract_features


def _feat(url):
    return dict(zip(FEATURE_NAMES, extract_features(url)))


def test_hex_like_hostname_is_not_ip():
    assert _feat("http://cafe/login")["is_ip_address"] == 0.0


def test_ipv6_detected():
    assert _feat("http://[2001:db8::1]/x")["is_ip_address"] == 1.0


def test_invalid_port_does_not_crash():
    f = _feat("http://example.com:99999/path")
    assert f["has_unusual_port"] == 1.0


def test_malformed_ipv6_does_not_crash():
    assert len(extract_features("http://[::1/path")) == len(FEATURE_NAMES)
