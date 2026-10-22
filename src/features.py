"""URL feature extraction for phishing detection.

Extracts purely lexical / structural features from a URL string.
No network requests are made.
"""

import re
from urllib.parse import urlparse, parse_qs

# Ordered list of feature names — must match the vector returned by extract_features.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "num_dots",
    "num_subdomains",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_at_symbols",
    "is_ip_address",
    "is_https",
    "num_path_separators",
    "num_query_params",
    "has_unusual_port",
]

# Ports considered "normal" for web traffic.
_STANDARD_PORTS = {80, 443, 8080, 8443, ""}

# Regex that matches an IPv4 address.
_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

# Regex that roughly matches an IPv6 address (including bracket-wrapped).
_IPV6_RE = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")

# Characters considered "special" in a URL (excluding dot, hyphen, digits, letters).
_SPECIAL_CHARS_RE = re.compile(r"[^a-zA-Z0-9.\-]")


def _safe_parse(url: str) -> urlparse:
    """Parse a URL, prepending a scheme if absent so urlparse works correctly."""
    url = url.strip()
    if not url:
        return urlparse("")
    # If no scheme, prepend http:// so urlparse can split correctly.
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def _is_ip_hostname(hostname: str) -> bool:
    """Return True if *hostname* looks like an IPv4 or IPv6 address."""
    if not hostname:
        return False
    return bool(_IPV4_RE.match(hostname)) or bool(_IPV6_RE.match(hostname))


def extract_features(url: str) -> list[float]:
    """Return a numerical feature vector for *url*.

    The vector length always equals ``len(FEATURE_NAMES)``.
    If the URL is empty or un-parseable, zeros are returned for most fields.
    """
    if not isinstance(url, str) or not url.strip():
        return [0.0] * len(FEATURE_NAMES)

    parsed = _safe_parse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    raw = url.strip()

    url_length = float(len(raw))
    hostname_length = float(len(hostname))
    num_dots = float(raw.count("."))

    # Subdomains: parts of hostname minus the registered domain + TLD.
    # e.g. "a.b.example.com" -> parts=["a","b","example","com"] -> subdomains = 2
    hostname_parts = hostname.split(".") if hostname else []
    num_subdomains = float(max(len(hostname_parts) - 2, 0))

    num_hyphens = float(raw.count("-"))
    num_digits = float(sum(c.isdigit() for c in raw))
    num_special = float(len(_SPECIAL_CHARS_RE.findall(raw)))
    num_at = float(raw.count("@"))
    is_ip = 1.0 if _is_ip_hostname(hostname) else 0.0
    is_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    # Path separators (exclude the leading /)
    num_path_sep = float(max(path.count("/") - 1, 0)) if path else 0.0

    # Query parameters
    try:
        num_qparams = float(len(parse_qs(query)))
    except Exception:
        num_qparams = 0.0

    # Unusual port
    port = parsed.port  # None when no explicit port
    if port is None:
        has_unusual_port = 0.0
    else:
        has_unusual_port = 0.0 if port in _STANDARD_PORTS else 1.0

    return [
        url_length,
        hostname_length,
        num_dots,
        num_subdomains,
        num_hyphens,
        num_digits,
        num_special,
        num_at,
        is_ip,
        is_https,
        num_path_sep,
        num_qparams,
        has_unusual_port,
    ]
