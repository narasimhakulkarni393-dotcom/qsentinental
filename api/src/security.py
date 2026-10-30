"""Q-Sentinel — Backend Security & SSRF Protection Module.

Ensures zero-network static URL inspection and guards against Server-Side Request Forgery (SSRF),
cloud metadata exfiltration, loopback abuse, and malformed request attacks.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

# Maximum allowed URL length to prevent ReDoS and memory exhaustion
MAX_URL_LENGTH = 4096

# Dangerous SSRF hostname literals
LOCALHOST_NAMES = {
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"
}

# AWS, GCP, Azure, OpenStack, DigitalOcean instance metadata endpoints
CLOUD_METADATA_IPS = {
    "169.254.169.254", "fd00:ec2::254"
}


def validate_and_sanitize_url(url: str) -> dict:
    """Validate, sanitize, and verify security boundaries for an untrusted input URL.

    Returns
    -------
    dict:
        - valid: bool
        - is_valid: bool
        - sanitized_url: str
        - hostname: str
        - ssrf_risk: bool
        - ssrf_type: str or None
        - error: str or None
    """
    if not isinstance(url, str):
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": "",
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": "URL must be a non-null string"
        }

    clean = url.strip()
    if not clean:
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": "",
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": "URL cannot be empty or whitespace only"
        }

    if len(clean) > MAX_URL_LENGTH:
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": "",
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": f"URL length ({len(clean)} chars) exceeds maximum allowable limit of {MAX_URL_LENGTH} characters"
        }

    # Control character check
    if any(ord(c) < 32 or ord(c) == 127 for c in clean):
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": "",
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": "URL contains illegal control or newline characters"
        }

    # Prepend default scheme if absent for safe parsing
    eval_url = clean
    if "://" not in eval_url:
        eval_url = "http://" + eval_url

    try:
        parsed = urlparse(eval_url)
        hostname = (parsed.hostname or "").strip().lower()
    except Exception as e:
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": clean,
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": f"Malformed URL syntax: {str(e)}"
        }

    if not hostname:
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": clean,
            "hostname": "",
            "ssrf_risk": False,
            "ssrf_type": None,
            "error": "URL does not contain a valid destination hostname or authority"
        }

    # SSRF / Hostname Analysis
    ssrf_risk = False
    ssrf_type = None

    if hostname in LOCALHOST_NAMES:
        ssrf_risk = True
        ssrf_type = "loopback_localhost"

    # Check IP Literals (IPv4 and IPv6)
    clean_ip = hostname.strip("[]")
    try:
        ip_obj = ipaddress.ip_address(clean_ip)
        if clean_ip in CLOUD_METADATA_IPS:
            ssrf_risk = True
            ssrf_type = "cloud_metadata_service"
        elif ip_obj.is_loopback:
            ssrf_risk = True
            ssrf_type = "loopback_ip"
        elif ip_obj.is_private:
            ssrf_risk = True
            ssrf_type = "private_rfc1918_network"
        elif ip_obj.is_link_local:
            ssrf_risk = True
            ssrf_type = "link_local_network"
        elif ip_obj.is_unspecified:
            ssrf_risk = True
            ssrf_type = "unspecified_ip"
    except ValueError:
        # Not an IP literal, it is a domain name
        pass

    if ssrf_risk:
        return {
            "valid": False,
            "is_valid": False,
            "sanitized_url": clean,
            "hostname": hostname,
            "ssrf_risk": True,
            "ssrf_type": ssrf_type,
            "error": f"Prohibited SSRF target: {ssrf_type} ({hostname})"
        }

    return {
        "valid": True,
        "is_valid": True,
        "sanitized_url": clean,
        "hostname": hostname,
        "ssrf_risk": False,
        "ssrf_type": None,
        "error": None
    }
