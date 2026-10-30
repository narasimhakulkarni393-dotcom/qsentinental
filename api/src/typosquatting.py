"""Q-Sentinel — Domain Impersonation, Typosquatting, and Homoglyph Detection Engine.

Provides evidence-based domain security analysis without remote network calls.
Detects:
- Numeric and leetspeak character substitutions (0->o, 1->l/i, 3->e, 5->s, rn->m, vv->w, etc.)
- Unicode homoglyph and IDN homograph attacks (e.g., Cyrillic 'о' for Latin 'o')
- Punycode encoded domains (xn--)
- Character repetition (gooogle.com) and omission (gogle.com)
- Character transpositions (googel.com)
- Damerau-Levenshtein edit distance against registered brand names
- Combosquatting (brand keywords combined with phishing/security lure words)
- Subdomain brand masquerading (brand in subdomain of an external domain)
- Synthetic / reserved TLDs (RFC 2606 / RFC 6761: .example, .invalid, .test, .localhost)
- Full exclusion for genuine, official brand domains
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# ==============================================================================
# Configurable Protected Brands Database
# ==============================================================================

DEFAULT_PROTECTED_BRANDS: dict[str, dict[str, Any]] = {
    "google": {
        "name": "Google",
        "domain": "google.com",
        "sld": "google",
        "aliases": {
            "google.com", "google.co.uk", "google.ca", "google.de", "google.fr",
            "google.it", "google.co.in", "google.co.jp", "google.com.au",
            "googlevideo.com", "googleapis.com", "gstatic.com", "youtube.com", "gmail.com"
        }
    },
    "microsoft": {
        "name": "Microsoft",
        "domain": "microsoft.com",
        "sld": "microsoft",
        "aliases": {
            "microsoft.com", "live.com", "office.com", "outlook.com", "msn.com",
            "azure.com", "windows.com", "microsoftonline.com", "office365.com", "bing.com"
        }
    },
    "apple": {
        "name": "Apple",
        "domain": "apple.com",
        "sld": "apple",
        "aliases": {
            "apple.com", "icloud.com", "me.com", "itunes.com", "apple-dns.net"
        }
    },
    "amazon": {
        "name": "Amazon",
        "domain": "amazon.com",
        "sld": "amazon",
        "aliases": {
            "amazon.com", "amazon.co.uk", "amazon.de", "amazon.co.jp", "amazon.ca",
            "amazon.fr", "amazon.it", "amazon.in", "aws.amazon.com", "media-amazon.com",
            "primevideo.com", "ssl-images-amazon.com"
        }
    },
    "paypal": {
        "name": "PayPal",
        "domain": "paypal.com",
        "sld": "paypal",
        "aliases": {
            "paypal.com", "paypal.me", "paypal-communication.com", "paypalobjects.com"
        }
    },
    "facebook": {
        "name": "Facebook",
        "domain": "facebook.com",
        "sld": "facebook",
        "aliases": {
            "facebook.com", "fb.com", "meta.com", "messenger.com", "facebookcorewwwi.onion"
        }
    },
    "instagram": {
        "name": "Instagram",
        "domain": "instagram.com",
        "sld": "instagram",
        "aliases": {
            "instagram.com", "cdninstagram.com"
        }
    },
    "netflix": {
        "name": "Netflix",
        "domain": "netflix.com",
        "sld": "netflix",
        "aliases": {
            "netflix.com", "nflxext.com", "nflximg.net", "netflixinvestor.com"
        }
    },
    "github": {
        "name": "GitHub",
        "domain": "github.com",
        "sld": "github",
        "aliases": {
            "github.com", "github.io", "githubassets.com", "githubusercontent.com", "git.io"
        }
    },
    "linkedin": {
        "name": "LinkedIn",
        "domain": "linkedin.com",
        "sld": "linkedin",
        "aliases": {
            "linkedin.com", "licdn.com"
        }
    },
    "twitter": {
        "name": "Twitter / X",
        "domain": "x.com",
        "sld": "x",
        "aliases": {
            "twitter.com", "x.com", "t.co", "twimg.com"
        }
    },
    "chase": {
        "name": "Chase Bank",
        "domain": "chase.com",
        "sld": "chase",
        "aliases": {
            "chase.com", "jpmorganchase.com", "chasepaymentech.com"
        }
    },
    "wellsfargo": {
        "name": "Wells Fargo",
        "domain": "wellsfargo.com",
        "sld": "wellsfargo",
        "aliases": {
            "wellsfargo.com"
        }
    },
    "bankofamerica": {
        "name": "Bank of America",
        "domain": "bankofamerica.com",
        "sld": "bankofamerica",
        "aliases": {
            "bankofamerica.com", "bofa.com"
        }
    }
}

# Unicode Homoglyphs: Map visually identical non-Latin scripts to standard Latin ASCII
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic
    'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'ѕ': 's', 'х': 'x', 'у': 'y',
    'і': 'i', 'ј': 'j', 'ӏ': 'l', 'ո': 'n', 'в': 'b', 'һ': 'h', 'ԁ': 'd', 'ԛ': 'q',
    'ԃ': 'd', 'ԍ': 'g', 'м': 'm', 'т': 't',
    # Greek
    'α': 'a', 'ο': 'o', 'ρ': 'p', 'ν': 'v', 'κ': 'k', 'τ': 't', 'γ': 'y', 'χ': 'x',
    'ι': 'i', 'ε': 'e',
    # Fullwidth / numeric
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
}

# Security and authentication lure words common in combosquatting attacks
SECURITY_LURE_WORDS: set[str] = {
    "security", "login", "signin", "verify", "verification", "account", "auth",
    "support", "update", "portal", "secure", "service", "billing", "help",
    "alert", "token", "password", "wallet", "bank", "banking", "oauth",
    "confirm", "protect", "client", "safe", "center", "system", "recover",
    "suspended", "unlock", "validation", "authenticate", "identity"
}

# Reserved synthetic TLDs per RFC 2606 and RFC 6761
RESERVED_TLDS: set[str] = {"example", "invalid", "localhost", "test", "local", "onion"}

# Multi-part national/country-code TLDs
TWO_PART_TLDS: set[str] = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk",
    "com.au", "net.au", "org.au", "edu.au",
    "co.in", "net.in", "org.in", "gen.in",
    "co.jp", "ne.jp", "ac.jp",
    "co.nz", "net.nz", "org.nz",
    "com.br", "net.br",
    "com.mx", "org.mx"
}


# ==============================================================================
# Helper Distance and Parsing Algorithms
# ==============================================================================

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein distance (insert, delete, substitute, transpose)."""
    d: dict[tuple[int, int], int] = {}
    len1 = len(s1)
    len2 = len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            # adjacent transposition check
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)
    return d[(len1 - 1, len2 - 1)]


def extract_domain_components(url_or_domain: str) -> dict[str, Any]:
    """Parse and decompose a URL or hostname into clean structural domain components."""
    raw = (url_or_domain or "").strip()
    if not raw:
        return {
            "hostname": "",
            "registered_domain": "",
            "sld": "",
            "tld": "",
            "subdomains": [],
            "is_punycode": False,
            "has_homoglyph": False,
            "is_reserved_tld": False,
            "error": "Empty input string"
        }

    if "://" not in raw:
        raw = "http://" + raw

    try:
        parsed = urlparse(raw)
        hostname = (parsed.hostname or "").strip().lower()
    except Exception:
        hostname = raw.split("/")[0].split(":")[0].strip().lower()

    # Punycode check and decoding
    is_punycode = False
    decoded_hostname = hostname
    if "xn--" in hostname:
        is_punycode = True
        try:
            decoded_hostname = hostname.encode("ascii").decode("idna")
        except Exception:
            decoded_hostname = hostname

    # Unicode homoglyph normalization
    normalized_hostname = "".join(HOMOGLYPH_MAP.get(c, c) for c in decoded_hostname)
    has_homoglyph = normalized_hostname != decoded_hostname

    parts = normalized_hostname.split(".") if normalized_hostname else []
    tld = ""
    sld = ""
    subdomains = []

    if len(parts) >= 2:
        last_two = ".".join(parts[-2:])
        if last_two in TWO_PART_TLDS and len(parts) >= 3:
            tld = last_two
            sld = parts[-3]
            subdomains = parts[:-3]
        else:
            tld = parts[-1]
            sld = parts[-2]
            subdomains = parts[:-2]
    elif len(parts) == 1:
        sld = parts[0]

    registered_domain = f"{sld}.{tld}" if tld else sld

    return {
        "hostname": hostname,
        "decoded_hostname": decoded_hostname,
        "normalized_hostname": normalized_hostname,
        "registered_domain": registered_domain,
        "sld": sld,
        "tld": tld,
        "subdomains": subdomains,
        "is_punycode": is_punycode,
        "has_homoglyph": has_homoglyph,
        "is_reserved_tld": tld in RESERVED_TLDS,
        "error": None
    }


def get_leet_candidates(text: str) -> list[tuple[str, list[str]]]:
    """Generate normalized strings by reversing leetspeak and numeric substitutions.

    Returns a list of tuples: (normalized_candidate_string, reasons_list).
    """
    t = text
    subs: list[str] = []

    # Sequence substitutions
    if "rn" in t:
        t = t.replace("rn", "m")
        subs.append("Character sequence substitution ('rn' for 'm')")
    if "vv" in t:
        t = t.replace("vv", "w")
        subs.append("Character sequence substitution ('vv' for 'w')")

    candidates: list[tuple[str, list[str]]] = [(t, list(subs))]

    # '0' -> 'o'
    if "0" in t:
        new_c = []
        for cand, s in candidates:
            new_c.append((cand.replace("0", "o"), s + ["Numeric character substitution ('0' for 'o')"]))
        candidates = new_c

    # '3' -> 'e'
    if "3" in t:
        new_c = []
        for cand, s in candidates:
            new_c.append((cand.replace("3", "e"), s + ["Numeric character substitution ('3' for 'e')"]))
        candidates = new_c

    # '5' -> 's'
    if "5" in t:
        new_c = []
        for cand, s in candidates:
            new_c.append((cand.replace("5", "s"), s + ["Numeric character substitution ('5' for 's')"]))
        candidates = new_c

    # '1' -> can replace 'l' or 'i'
    if "1" in t:
        new_c = []
        for cand, s in candidates:
            new_c.append((cand.replace("1", "l"), s + ["Numeric character substitution ('1' for 'l')"]))
            new_c.append((cand.replace("1", "i"), s + ["Numeric character substitution ('1' for 'i')"]))
        candidates = new_c

    return candidates


# ==============================================================================
# Core Typosquatting & Brand Impersonation Detector
# ==============================================================================

def check_domain_impersonation(
    url_or_domain: str,
    custom_brands: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Analyze a URL or domain string for typosquatting and brand impersonation.

    Parameters
    ----------
    url_or_domain : str
        Target URL or hostname to inspect.
    custom_brands : dict, optional
        Additional brand database entries to include or override.

    Returns
    -------
    dict
        Structured analysis containing:
        - detected: bool
        - target_brand: str or None
        - legitimate_domain: str or None
        - is_genuine: bool
        - similarity_score: float
        - reasons: list of str
        - detection_engine: list of str
        - is_reserved_tld: bool
        - classification_hint: str ("SAFE", "SUSPICIOUS", "UNKNOWN")
    """
    brands = dict(DEFAULT_PROTECTED_BRANDS)
    if custom_brands:
        brands.update(custom_brands)

    parts = extract_domain_components(url_or_domain)
    if parts.get("error"):
        return {
            "detected": False,
            "target_brand": None,
            "legitimate_domain": None,
            "is_genuine": False,
            "similarity_score": 0.0,
            "reasons": [parts["error"]],
            "detection_engine": [],
            "is_reserved_tld": False,
            "classification_hint": "SCAN_ERROR"
        }

    hostname = parts["hostname"]
    registered_domain = parts["registered_domain"]
    sld = parts["sld"]
    tld = parts["tld"]
    subdomains = parts["subdomains"]

    # 1. Genuine Official Domain Verification
    # Must NOT contain homoglyphs or punycode to be genuinely the brand's official domain
    if not parts["has_homoglyph"] and not parts["is_punycode"]:
        for b_key, b_info in brands.items():
            official_domain = b_info["domain"]
            aliases = b_info.get("aliases", {official_domain})
            if registered_domain == official_domain or registered_domain in aliases or hostname == official_domain:
                return {
                    "detected": False,
                    "target_brand": b_info["name"],
                    "legitimate_domain": b_info["domain"],
                    "is_genuine": True,
                    "similarity_score": 1.0,
                    "reasons": [f"Verified legitimate domain for {b_info['name']}"],
                    "detection_engine": [],
                    "is_reserved_tld": False,
                    "classification_hint": "SAFE"
                }

    # 2. Impersonation & Typosquatting Analysis against Protected Brands
    engines: set[str] = set()
    reasons: list[str] = []
    target_brand: str | None = None
    legit_domain: str | None = None
    similarity_score = 0.0

    for b_key, b_info in brands.items():
        b_name = b_info["name"]
        b_domain = b_info["domain"]
        b_sld = b_info["sld"]

        # A. Homoglyph / Punycode check
        if parts["has_homoglyph"] or parts["is_punycode"]:
            if parts["sld"] == b_sld:
                target_brand = b_name
                legit_domain = b_domain
                engines.update(["typosquatting", "brand_impersonation", "domain_similarity"])
                reasons.append(f"Possible {b_name} impersonation")
                reasons.append("Unicode homoglyph characters detected in domain")
                similarity_score = 0.99
                break

        # B. Leetspeak & Numeric Character Substitutions
        leet_candidates = get_leet_candidates(sld)
        matched_leet = False
        for leet_cand, sub_reasons in leet_candidates:
            if leet_cand == b_sld:
                target_brand = b_name
                legit_domain = b_domain
                engines.update(["typosquatting", "brand_impersonation", "domain_similarity"])
                reasons.append(f"Possible {b_name} impersonation")
                reasons.append("Numeric character substitution detected")
                for sr in sub_reasons:
                    if sr not in reasons:
                        reasons.append(sr)
                similarity_score = 0.95
                matched_leet = True
                break
        if matched_leet:
            break

        # C. Combosquatting: Protected brand combined with lure tokens or hyphens in SLD
        # e.g. github-security-login.example, paypal-account-verify.com
        tokens = set(re.split(r"[-_.]", sld))
        lure_matches = tokens.intersection(SECURITY_LURE_WORDS)
        brand_matches_combosquat = (b_sld in tokens) or (len(b_sld) >= 4 and b_sld in sld and sld != b_sld)
        if brand_matches_combosquat:
            if lure_matches or (len(tokens) > 1 and b_sld in tokens):
                target_brand = b_name
                legit_domain = b_domain
                engines.update(["typosquatting", "brand_impersonation", "combosquatting"])
                reasons.append(f"Possible {b_name} impersonation")
                if lure_matches:
                    reasons.append(f"Brand keyword '{b_name}' combined with security lure tokens ({', '.join(sorted(lure_matches))})")
                else:
                    reasons.append(f"Brand keyword '{b_name}' combined in hyphenated domain '{sld}'")
                similarity_score = 0.90
                break

        # D. Subdomain Masquerading: Brand placed in subdomain of an external domain
        # e.g. google.com.login-portal.example
        subdomain_match = False
        if len(b_sld) >= 3:
            subdomain_match = b_sld in subdomains or any(b_sld in sub for sub in subdomains)
        else:
            subdomain_match = b_sld in subdomains
        if subdomain_match:
            target_brand = b_name
            legit_domain = b_domain
            engines.update(["brand_impersonation", "subdomain_spoofing"])
            reasons.append(f"Possible {b_name} impersonation")
            reasons.append(f"Protected brand '{b_name}' masqueraded in subdomain hierarchy of external domain '{registered_domain}'")
            similarity_score = 0.88
            break

        # E. Damerau-Levenshtein Edit Distance (Repetition, Omission, Transposition)
        if len(sld) >= 3 and len(b_sld) >= 3:
            dist = damerau_levenshtein_distance(sld, b_sld)
            # Threshold: distance 1 is high confidence typosquatting
            if dist == 1:
                target_brand = b_name
                legit_domain = b_domain
                engines.update(["typosquatting", "brand_impersonation", "domain_similarity"])
                reasons.append(f"Possible {b_name} impersonation")
                if len(sld) > len(b_sld):
                    reasons.append("Repeated character typosquatting detected")
                elif len(sld) < len(b_sld):
                    reasons.append("Missing character typosquatting detected")
                elif sorted(sld) == sorted(b_sld):
                    reasons.append("Adjacent character transposition typosquatting detected")
                else:
                    reasons.append("Single-character substitution typosquatting detected")
                similarity_score = round(1.0 - (dist / max(len(sld), len(b_sld))), 2)
                break

    # 3. Check for Reserved / Synthetic TLDs (RFC 2606)
    if parts["is_reserved_tld"]:
        if target_brand:
            reasons.append(f"Reserved synthetic TLD (.{tld}) used with brand impersonation")
        else:
            reasons.append(f"Reserved synthetic domain (.{tld}) cannot be resolved on public DNS")

    detected = target_brand is not None

    # Classification hint for orchestration
    if detected:
        classification_hint = "SUSPICIOUS"
    elif parts["is_reserved_tld"]:
        classification_hint = "UNKNOWN"
    else:
        classification_hint = "SAFE"

    return {
        "detected": detected,
        "target_brand": target_brand,
        "legitimate_domain": legit_domain,
        "is_genuine": False,
        "similarity_score": similarity_score,
        "reasons": reasons,
        "detection_engine": sorted(list(engines)),
        "is_reserved_tld": parts["is_reserved_tld"],
        "classification_hint": classification_hint
    }
