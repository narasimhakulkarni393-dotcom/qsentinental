"""URL feature extraction for phishing detection in Q-Sentinel.

Extracts purely lexical and structural features from a URL string without making
any network, DNS, or HTTP requests (zero-network inspection prevents SSRF,
DNS-rebinding, and remote code execution risks).
"""

import math
import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

# Ordered list of feature names — must match the vector returned by extract_features.
FEATURE_NAMES = [
    # Basic lengths and counts
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_subdomains",
    "num_hyphens",
    "hyphen_in_domain",
    "num_digits",
    "digit_ratio_url",
    "digit_ratio_domain",
    "num_special_chars",
    "num_at_symbols",
    # Protocol & host indicators
    "is_ip_address",
    "is_https",
    "has_unusual_port",
    "has_punycode",
    # Lexical patterns & structural obfuscation
    "num_path_separators",
    "num_query_params",
    "has_double_slash_path",
    "entropy_url",
    "entropy_domain",
    "has_suspicious_tld",
    "num_sensitive_keywords",
    "has_brand_in_subdomain",
]

# Ports considered "normal" for web traffic.
_STANDARD_PORTS = {80, 443, 8080, 8443}

# Regex for IPv4 address.
_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

# Regex that roughly matches an IPv6 address.
_IPV6_RE = re.compile(r"^\[?[0-9a-fA-F]*:[0-9a-fA-F:.]*\]?$")

# Characters considered "special" in a URL (excluding dot, hyphen, digits, letters).
_SPECIAL_CHARS_RE = re.compile(r"[^a-zA-Z0-9.\-]")

# TLDs historically associated with free registration and elevated phishing/abuse ratios.
_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "buzz", "fit", "icu",
    "vip", "click", "rest", "country", "stream", "date", "download", "racing",
    "win", "bid", "loan", "men", "party", "trade", "accountant", "faith", "review",
    "cricket", "kim", "science", "gdn", "mom", "monster", "hair", "beauty", "quest"
}

# Targeted lure/authentication keywords heavily exploited in credential phishing campaigns.
_SENSITIVE_KEYWORDS = [
    "login", "signin", "verify", "verification", "update", "security", "secure",
    "banking", "bank", "account", "password", "passwd", "credential", "wallet",
    "admin", "confirm", "confirmation", "auth", "authorize", "recover", "recovery",
    "billing", "invoice", "ebayisapi", "webscr", "appleid", "support", "service",
    "session", "token", "unlock", "alert", "notice"
]

# High-profile brand tokens commonly used in subdomain deception.
_POPULAR_BRANDS = [
    "paypal", "google", "apple", "microsoft", "netflix", "amazon", "facebook",
    "instagram", "whatsapp", "dropbox", "chase", "wellsfargo", "bankofamerica",
    "citibank", "coinbase", "binance", "metamask", "steam", "linkedin", "yahoo",
    "outlook", "office365", "dhl", "fedex", "usps"
]


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string (in bits per character)."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def _safe_parse(url: str):
    """Parse a URL safely, prepending a scheme if absent so urlparse extracts components correctly."""
    url = url.strip()
    if not url:
        return urlparse("")
    # If no scheme, prepend http:// so urlparse parses hostname vs path correctly.
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def _is_ip_hostname(hostname: str) -> bool:
    """Return True if hostname matches an IPv4 or IPv6 pattern."""
    if not hostname:
        return False
    clean = hostname.strip("[]")
    return bool(_IPV4_RE.match(clean)) or (":" in hostname and bool(_IPV6_RE.match(hostname)))


def extract_features(url: str) -> list[float]:
    """Return a fixed numerical feature vector for *url*.

    The vector length strictly equals ``len(FEATURE_NAMES)``.
    Operates in sub-millisecond time and handles malformed or non-ASCII inputs safely.
    """
    if not isinstance(url, str) or not url.strip():
        return [0.0] * len(FEATURE_NAMES)

    raw = url.strip()

    try:
        parsed = _safe_parse(raw)
        raw_host = (parsed.hostname or "").lower()
    except ValueError:
        parsed = None
        raw_host = ""

    # Normalize out www. prefix from host for consistent evaluation between apex & www domains
    hostname = raw_host[4:] if raw_host.startswith("www.") else raw_host

    path = (parsed.path or "") if parsed else ""
    query = (parsed.query or "") if parsed else ""
    scheme = (parsed.scheme or "").lower() if parsed else ""

    # Normalized URL for consistent length, dots, and entropy
    clean_url = f"{scheme}://{hostname}{path}{('?' + query) if query else ''}"
    raw_lower = raw.lower()

    # 1. Lengths
    url_length = float(len(clean_url))
    hostname_length = float(len(hostname))
    path_length = float(len(path))
    query_length = float(len(query))

    # 2. Dot and Subdomain counts
    num_dots = float(clean_url.count("."))
    hostname_parts = hostname.split(".") if hostname else []
    # Subdomains = parts minus registered domain name + TLD (excluding any www)
    subdomain_parts = [p for p in hostname_parts[:-2] if p != "www"] if len(hostname_parts) >= 2 else []
    num_subdomains = float(len(subdomain_parts))

    # 3. Hyphens & domain hyphens
    num_hyphens = float(raw.count("-"))
    hyphen_in_domain = float(hostname.count("-"))

    # 4. Digits and ratios
    num_digits = float(sum(c.isdigit() for c in raw))
    digit_ratio_url = num_digits / url_length if url_length > 0 else 0.0
    domain_digits = sum(c.isdigit() for c in hostname)
    digit_ratio_domain = float(domain_digits / hostname_length) if hostname_length > 0 else 0.0

    # 5. Characters and symbols
    num_special = float(len(_SPECIAL_CHARS_RE.findall(raw)))
    num_at = float(raw.count("@"))

    # 6. Host & Protocol indicators
    is_ip = 1.0 if _is_ip_hostname(hostname) else 0.0
    is_https = 1.0 if scheme == "https" else 0.0

    # Unusual port
    try:
        port = parsed.port if parsed else None
    except ValueError:
        port = -1
    if port is None:
        has_unusual_port = 0.0
    else:
        has_unusual_port = 0.0 if port in _STANDARD_PORTS else 1.0

    # Punycode / Homograph (IDN) check
    has_punycode = 1.0 if ("xn--" in hostname or any(ord(c) > 127 for c in hostname)) else 0.0

    # 7. Structural path & query analysis
    num_path_separators = float(max(path.count("/") - 1, 0)) if path else 0.0
    try:
        num_query_params = float(len(parse_qs(query)))
    except Exception:
        num_query_params = 0.0
    has_double_slash_path = 1.0 if "//" in path else 0.0

    # 8. Entropy
    entropy_url = float(_shannon_entropy(clean_url))
    entropy_domain = float(_shannon_entropy(hostname))

    # 9. TLD analysis
    tld = hostname_parts[-1] if len(hostname_parts) >= 2 else ""
    has_suspicious_tld = 1.0 if tld in _SUSPICIOUS_TLDS else 0.0

    # 10. Sensitive keyword frequency
    keyword_hits = sum(1 for kw in _SENSITIVE_KEYWORDS if kw in raw_lower)
    num_sensitive_keywords = float(keyword_hits)

    # 11. Brand spoofing in subdomains:
    # If a known brand is present in hostname, but the registered domain is not that brand
    has_brand_in_sub = 0.0
    if len(hostname_parts) >= 3:
        registered_domain = hostname_parts[-2].lower()
        subdomain_part = ".".join(subdomain_parts).lower()
        for brand in _POPULAR_BRANDS:
            if brand in subdomain_part and brand != registered_domain:
                has_brand_in_sub = 1.0
                break

    return [
        url_length,
        hostname_length,
        path_length,
        query_length,
        num_dots,
        num_subdomains,
        num_hyphens,
        hyphen_in_domain,
        num_digits,
        digit_ratio_url,
        digit_ratio_domain,
        num_special,
        num_at,
        is_ip,
        is_https,
        has_unusual_port,
        has_punycode,
        num_path_separators,
        num_query_params,
        has_double_slash_path,
        entropy_url,
        entropy_domain,
        has_suspicious_tld,
        num_sensitive_keywords,
        has_brand_in_sub,
    ]
