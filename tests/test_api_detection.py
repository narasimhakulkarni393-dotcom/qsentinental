import pytest
from api.index import perform_full_scan, get_threat_intelligence


def test_g00gle_typosquatting_exact_schema():
    """Verify that https://www.g00gle.com produces the exact required schema and is SUSPICIOUS."""
    res = perform_full_scan("https://www.g00gle.com")
    
    assert res["url"] == "https://www.g00gle.com"
    assert res["classification"] == "SUSPICIOUS"
    assert res["risk_score"] == 85
    assert res["confidence"] is None or isinstance(res["confidence"], float)
    assert "typosquatting" in res["detection_engine"]
    assert "brand_impersonation" in res["detection_engine"]
    assert res["brand_impersonation"]["detected"] is True
    assert res["brand_impersonation"]["target_brand"] == "Google"
    assert res["brand_impersonation"]["legitimate_domain"] == "google.com"
    assert "Possible Google impersonation" in res["reasons"]
    assert "Numeric character substitution detected" in res["reasons"]
    assert res["model_status"] in ("success", "fallback")
    
    # Backward compatibility fields
    assert res["prediction"] == "phishing"
    assert res["risk_level"] in ("high", "critical")
    assert "probabilities" in res


@pytest.mark.parametrize("url,expected_brand", [
    ("paypa1.com", "PayPal"),
    ("amaz0n.com", "Amazon"),
    ("micr0soft.com", "Microsoft"),
    ("app1e.com", "Apple"),
    ("faceb00k.com", "Facebook"),
    ("netf1ix.com", "Netflix"),
    ("github-security-login.example", "GitHub"),
])
def test_suspicious_lookalike_domains(url, expected_brand):
    res = perform_full_scan(url)
    assert res["classification"] == "SUSPICIOUS"
    assert res["risk_score"] >= 80
    assert res["brand_impersonation"]["detected"] is True
    assert res["brand_impersonation"]["target_brand"] == expected_brand
    assert res["prediction"] == "phishing"


@pytest.mark.parametrize("url,expected_brand", [
    ("google.com", "Google"),
    ("https://www.google.com", "Google"),
    ("microsoft.com", "Microsoft"),
    ("apple.com", "Apple"),
    ("amazon.com", "Amazon"),
    ("paypal.com", "PayPal"),
    ("github.com", "GitHub"),
])
def test_legitimate_domains_are_safe(url, expected_brand):
    res = perform_full_scan(url)
    assert res["classification"] == "SAFE"
    assert res["risk_score"] <= 15
    assert res["brand_impersonation"]["detected"] is False
    assert res["prediction"] == "legitimate"


def test_test_example_is_unknown():
    res = perform_full_scan("https://test.example")
    assert res["classification"] == "UNKNOWN"
    assert res["brand_impersonation"]["detected"] is False
    assert res["prediction"] == "unknown"


@pytest.mark.parametrize("ssrf_url", [
    "http://169.254.169.254/latest/meta-data",
    "http://127.0.0.1:8000",
    "http://localhost:5000",
    "http://10.0.0.1/admin",
    "http://192.168.1.1",
    "http://[::1]/",
])
def test_ssrf_prohibited_urls(ssrf_url):
    res = perform_full_scan(ssrf_url)
    assert res["classification"] == "SCAN_ERROR"
    assert res["input_valid"] is False
    assert res["model_status"] == "error"
    assert "SSRF" in res["reasons"][0] or "loopback" in res["reasons"][0] or "Prohibited" in res["reasons"][0]


def test_malformed_urls():
    for malformed in ["", "   ", "http://", "htp://::bad"]:
        res = perform_full_scan(malformed)
        assert res["classification"] == "SCAN_ERROR"
        assert res["input_valid"] is False


def test_model_failure_never_returns_safe(monkeypatch):
    """Verify that when ML models fail to load or infer, fallback is UNKNOWN, never SAFE."""
    import api.index as idx
    monkeypatch.setattr(idx, "get_models", lambda: (False, False))
    monkeypatch.setattr(idx, "predict_url", None)
    
    # An unknown generic domain without brand signals
    res = idx.perform_full_scan("https://some-unseen-generic-domain-xyz123.org")
    assert res["classification"] == "UNKNOWN"
    assert res["classification"] != "SAFE"
    assert res["model_status"] == "error"
