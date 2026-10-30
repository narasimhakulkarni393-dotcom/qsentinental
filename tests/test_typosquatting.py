import pytest
from src.typosquatting import (
    damerau_levenshtein_distance,
    extract_domain_components,
    get_leet_candidates,
    check_domain_impersonation,
    DEFAULT_PROTECTED_BRANDS,
)


def test_damerau_levenshtein_distance():
    assert damerau_levenshtein_distance("google", "google") == 0
    # 1 insertion
    assert damerau_levenshtein_distance("google", "googlle") == 1
    # 1 deletion
    assert damerau_levenshtein_distance("google", "gogle") == 1
    # 1 substitution
    assert damerau_levenshtein_distance("google", "goog1e") == 1
    # 1 transposition
    assert damerau_levenshtein_distance("google", "goolge") == 1


def test_extract_domain_components():
    res1 = extract_domain_components("https://www.g00gle.com/path?q=1")
    assert res1["hostname"] == "www.g00gle.com"
    assert res1["registered_domain"] == "g00gle.com"
    assert res1["sld"] == "g00gle"
    assert res1["tld"] == "com"

    # Multi-part TLD
    res2 = extract_domain_components("http://login.paypal.co.uk/auth")
    assert res2["registered_domain"] == "paypal.co.uk"
    assert res2["sld"] == "paypal"
    assert res2["tld"] == "co.uk"

    # Reserved synthetic TLD
    res3 = extract_domain_components("http://test.example")
    assert res3["is_reserved_tld"] is True
    assert res3["registered_domain"] == "test.example"


def test_leet_candidates():
    # 0 -> o
    cand_strs = [c for c, _ in get_leet_candidates("g00gle")]
    assert "google" in cand_strs

    # 1 -> l and i
    cand_strs = [c for c, _ in get_leet_candidates("paypa1")]
    assert "paypal" in cand_strs

    # 3 -> e
    cand_strs = [c for c, _ in get_leet_candidates("appl3")]
    assert "apple" in cand_strs

    # 5 -> s
    cand_strs = [c for c, _ in get_leet_candidates("micro5oft")]
    assert "microsoft" in cand_strs

    # rn -> m
    cand_strs = [c for c, _ in get_leet_candidates("arnazon")]
    assert "amazon" in cand_strs

    # vv -> w
    cand_strs = [c for c, _ in get_leet_candidates("vviki")]
    assert "wiki" in cand_strs


def test_numeric_substitutions_detected():
    cases = [
        ("https://www.g00gle.com", "Google", "google.com"),
        ("paypa1.com", "PayPal", "paypal.com"),
        ("amaz0n.com", "Amazon", "amazon.com"),
        ("micr0soft.com", "Microsoft", "microsoft.com"),
        ("app1e.com", "Apple", "apple.com"),
        ("faceb00k.com", "Facebook", "facebook.com"),
        ("netf1ix.com", "Netflix", "netflix.com"),
    ]

    for url, brand, domain in cases:
        res = check_domain_impersonation(url)
        assert res["detected"] is True, f"Failed for {url}"
        assert res["target_brand"] == brand
        assert res["legitimate_domain"] == domain
        assert "Possible " in res["reasons"][0]
        assert "Numeric character substitution detected" in res["reasons"]


def test_combosquatting_detected():
    url = "https://github-security-login.example/auth"
    res = check_domain_impersonation(url)
    assert res["detected"] is True
    assert res["target_brand"] == "GitHub"
    assert res["legitimate_domain"] == "github.com"
    assert res["classification_hint"] == "SUSPICIOUS"
    assert any("combosquatting" in eng for eng in res["detection_engine"])


def test_subdomain_spoofing_detected():
    url = "https://paypal.com.account-update.xyz/login"
    res = check_domain_impersonation(url)
    assert res["detected"] is True
    assert res["target_brand"] == "PayPal"
    assert "subdomain_spoofing" in res["detection_engine"]


def test_homoglyph_and_punycode_detected():
    # Cyrillic small letter a (U+0430) in apple.com
    cyrillic_apple = "https://\u0430pple.com"
    res = check_domain_impersonation(cyrillic_apple)
    assert res["detected"] is True
    assert res["target_brand"] == "Apple"


def test_genuine_protected_brands_pass():
    legit_domains = [
        "https://www.google.com",
        "google.com",
        "https://account.microsoft.com",
        "https://www.apple.com",
        "https://amazon.com",
        "https://paypal.com",
        "https://github.com",
        "https://linkedin.com",
        "https://netflix.com",
        "https://facebook.com",
    ]

    for d in legit_domains:
        res = check_domain_impersonation(d)
        assert res["detected"] is False, f"False positive on legitimate domain: {d}"
        assert res["is_genuine"] is True, f"Should be recognized as genuine: {d}"
        assert res["classification_hint"] == "SAFE"


def test_reserved_generic_tld():
    res = check_domain_impersonation("https://test.example")
    assert res["detected"] is False
    assert res["is_genuine"] is False
    assert res["is_reserved_tld"] is True
    assert res["classification_hint"] == "UNKNOWN"
