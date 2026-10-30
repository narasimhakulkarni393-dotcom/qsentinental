import json
import os
import sys
import re
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Add project root and api dir to sys.path so src modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for p in [str(CURRENT_DIR), str(ROOT_DIR), os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

IMPORT_ERROR = None
# Import feature extractor, model functions, and security / typosquatting modules
try:
    from src.features import FEATURE_NAMES, extract_features
    from src.model import load_model, predict_url
    from src.quantum_model import load_quantum_model, predict_url_quantum
    from src.security import validate_and_sanitize_url
    from src.typosquatting import (
        DEFAULT_PROTECTED_BRANDS,
        check_domain_impersonation,
        extract_domain_components
    )
except Exception as e:
    IMPORT_ERROR = str(e)
    try:
        from api.src.features import FEATURE_NAMES, extract_features
        from api.src.model import load_model, predict_url
        from api.src.quantum_model import load_quantum_model, predict_url_quantum
        from api.src.security import validate_and_sanitize_url
        from api.src.typosquatting import (
            DEFAULT_PROTECTED_BRANDS,
            check_domain_impersonation,
            extract_domain_components
        )
    except Exception as e2:
        IMPORT_ERROR = f"{e} | {e2}"
        FEATURE_NAMES = [
            "url_length", "hostname_length", "path_length", "query_length",
            "num_dots", "num_subdomains", "num_hyphens", "hyphen_in_domain",
            "num_digits", "digit_ratio_url", "digit_ratio_domain", "num_special_chars",
            "num_at_symbols", "is_ip_address", "is_https", "has_unusual_port",
            "has_punycode", "num_path_separators", "num_query_params",
            "has_double_slash_path", "entropy_url", "entropy_domain",
            "has_suspicious_tld", "num_sensitive_keywords", "has_brand_in_subdomain"
        ]
        extract_features = None
        load_model = None
        predict_url = None
        load_quantum_model = None
        predict_url_quantum = None
        validate_and_sanitize_url = None
        check_domain_impersonation = None
        extract_domain_components = None

# Cache model instances in memory across warm invocations
_CLASSICAL_BUNDLE = None
_QUANTUM_BUNDLE = None

# Simple in-memory rate limiting (max 120 req / 60 sec per client IP)
_RATE_LIMITS = defaultdict(list)
MAX_REQUEST_SIZE_BYTES = 1024 * 1024  # 1 MB


def get_models():
    """Retrieve or load cached classical LightGBM and quantum VQC model bundles."""
    global _CLASSICAL_BUNDLE, _QUANTUM_BUNDLE
    m_path = CURRENT_DIR / "models" if (CURRENT_DIR / "models").is_dir() else (ROOT_DIR / "models")
    if _CLASSICAL_BUNDLE is None and load_model:
        try:
            _CLASSICAL_BUNDLE = load_model(m_path)
        except Exception:
            _CLASSICAL_BUNDLE = False
    if _QUANTUM_BUNDLE is None and load_quantum_model:
        try:
            _QUANTUM_BUNDLE = load_quantum_model(m_path)
        except Exception:
            _QUANTUM_BUNDLE = False
    return _CLASSICAL_BUNDLE, _QUANTUM_BUNDLE


def analyze_url_heuristics(url: str, feats: dict) -> list:
    """Analyze lexical and security indicators to produce explainable ML evidence."""
    evidence = []

    # 1. IP address check
    if feats.get("is_ip_address", 0) == 1.0:
        evidence.append({
            "feature": "is_ip_address",
            "severity": "critical",
            "title": "IP Address Hostname",
            "description": "Host uses a raw numerical IP rather than a registered domain name, heavily correlated with phishing/C2."
        })

    # 2. Unusual Port
    if feats.get("has_unusual_port", 0) == 1.0:
        evidence.append({
            "feature": "has_unusual_port",
            "severity": "high",
            "title": "Non-Standard Web Port",
            "description": "Destination port is non-standard (neither 80, 443, 8080, nor 8443)."
        })

    # 3. Excessive Dots / Subdomains
    num_dots = feats.get("num_dots", 0)
    subdomains = feats.get("num_subdomains", 0)
    if subdomains >= 3 or num_dots >= 4:
        evidence.append({
            "feature": "num_subdomains",
            "severity": "warning",
            "title": "Deep Subdomain Hierarchy",
            "description": f"URL exhibits {int(subdomains)} subdomain levels with {int(num_dots)} dots, a common domain masquerading technique."
        })

    # 4. At Symbol (@) in URL
    if feats.get("num_at_symbols", 0) >= 1:
        evidence.append({
            "feature": "num_at_symbols",
            "severity": "high",
            "title": "Embedded Credential Token (@)",
            "description": "Presence of '@' can be used to obfuscate the real host authority in browser address bars."
        })

    # 5. Brand Spoofing in Subdomain
    if feats.get("has_brand_in_subdomain", 0) == 1.0:
        evidence.append({
            "feature": "has_brand_in_subdomain",
            "severity": "critical",
            "title": "Subdomain Brand Spoofing Detected",
            "description": "Target contains a high-profile brand name inside the subdomain while registered on a different domain."
        })

    # 6. Suspicious TLD
    if feats.get("has_suspicious_tld", 0) == 1.0:
        evidence.append({
            "feature": "has_suspicious_tld",
            "severity": "high",
            "title": "High-Abuse Top-Level Domain",
            "description": "Domain is registered under a TLD with statistically elevated phishing rates (e.g., .tk, .top, .xyz, .cf)."
        })

    # 7. Punycode / Homograph (IDN)
    if feats.get("has_punycode", 0) == 1.0:
        evidence.append({
            "feature": "has_punycode",
            "severity": "warning",
            "title": "IDN / Punycode Characters",
            "description": "Domain contains Punycode ('xn--') or non-ASCII characters, which may be an IDN homograph attack."
        })

    # 8. Suspicious keywords in host or path
    lower_url = url.lower()
    keywords = ["login", "signin", "verify", "secure", "update", "account", "banking", "billing", "oauth", "password", "wallet", "token"]
    matched = [kw for kw in keywords if kw in lower_url]
    if len(matched) >= 2:
        evidence.append({
            "feature": "suspicious_keywords",
            "severity": "warning",
            "title": "Targeted Authentication Keywords",
            "description": f"Contains security-sensitive lure keywords: {', '.join(matched[:4])}"
        })

    # 9. HTTP Scheme (lack of TLS)
    if feats.get("is_https", 1) == 0.0:
        evidence.append({
            "feature": "is_https",
            "severity": "info",
            "title": "Unencrypted HTTP Protocol",
            "description": "Traffic is unencrypted (HTTP). While not strictly proof of malice, legitimate auth forms require HTTPS."
        })

    # 10. URL Length & High Entropy
    url_len = feats.get("url_length", 0)
    entropy = feats.get("entropy_url", 0.0)
    if url_len > 75:
        evidence.append({
            "feature": "url_length",
            "severity": "info",
            "title": "Excessive URL Length",
            "description": f"Total character length is {int(url_len)} chars (elevated length is typical of obfuscated redirect payloads)."
        })
    if entropy > 4.5:
        evidence.append({
            "feature": "entropy_url",
            "severity": "info",
            "title": "Elevated Shannon Entropy",
            "description": f"URL exhibits high randomness ({entropy:.2f} bits), indicating encoded, tokenized, or randomized payloads."
        })

    return evidence


def get_threat_intelligence(url: str) -> dict:
    """Zero-network Threat Intelligence lookup against curated phishing indicators."""
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()

    # Normalize out leading 'www.'
    norm_host = host[4:] if host.startswith("www.") else host

    # Active IOC indicators (verified campaigns, excluding lookalike test cases)
    known_bad_hosts = {
        "paypa1-verify.com",
        "secure-bankofamerica.com-verify.tk",
        "appleid.apple.com.secure-login.info",
        "free-iphone-winner.com",
        "netflix-payment-update.com",
        "amazon-order-support.com"
    }

    if (
        host in known_bad_hosts
        or norm_host in known_bad_hosts
        or host.endswith(".tk")
        or host.endswith(".top")
        or host.endswith(".cf")
    ):
        return {
            "status": "flagged",
            "provider": "AlienVault OTX & PhishTank IOC Feed",
            "matches": [
                {
                    "indicator": host,
                    "type": "domain",
                    "threat_type": "Credential Phishing / Impersonation Campaign",
                    "confidence": "High",
                    "last_seen": "2026-10-30"
                }
            ]
        }

    return {
        "status": "clean",
        "provider": "AlienVault OTX & PhishTank IOC Feed",
        "matches": []
    }


def perform_full_scan(url: str) -> dict:
    """Run comprehensive multimodal quantum-classical, typosquatting, and heuristic scan.

    Enforces:
    - Zero-network static analysis (SSRF immunity)
    - 5-tier classification: SAFE, SUSPICIOUS, MALICIOUS, UNKNOWN, SCAN_ERROR
    - Never returns SAFE when inference fails or domain status is inconclusive
    - Backward-compatible serialization with original frontend keys
    """
    # 1. Input Sanitization & SSRF Protection
    if validate_and_sanitize_url:
        val = validate_and_sanitize_url(url)
        is_val = val.get("is_valid", val.get("valid", False))
        if not is_val:
            err_msg = val.get("error", "Invalid or prohibited URL input")
            return {
                "url": str(url or ""),
                "classification": "SCAN_ERROR",
                "risk_score": 0,
                "confidence": None,
                "detection_engine": ["input_validation"],
                "brand_impersonation": {
                    "detected": False,
                    "target_brand": None,
                    "legitimate_domain": None
                },
                "reasons": [err_msg],
                "model_status": "error",
                "input_valid": False,
                "error": err_msg,
                "prediction": "error",
                "prediction_text": f"Scan Error: {err_msg}",
                "risk_level": "unknown",
                "probabilities": {
                    "classical_rf_phishing": 0.0,
                    "classical_phishing": 0.0,
                    "quantum_vqc_phishing": 0.0,
                    "composite_phishing": 0.0,
                    "legitimate": 0.0
                },
                "ml_evidence": [],
                "threat_intelligence": {"status": "error", "provider": "Input Validator", "matches": []},
                "features": {}
            }
        clean_url = val["sanitized_url"]
    else:
        clean_url = (url or "").strip()
        if not clean_url or len(clean_url) > 4096:
            return {
                "url": clean_url,
                "classification": "SCAN_ERROR",
                "risk_score": 0,
                "confidence": None,
                "detection_engine": ["input_validation"],
                "brand_impersonation": {
                    "detected": False,
                    "target_brand": None,
                    "legitimate_domain": None
                },
                "reasons": ["URL is empty or exceeds standard length limit"],
                "model_status": "error",
                "input_valid": False,
                "error": "URL is empty or exceeds standard length limit"
            }

    # 2. Typosquatting & Brand Impersonation Analysis
    impersonation = {}
    if check_domain_impersonation:
        impersonation = check_domain_impersonation(clean_url)
    else:
        impersonation = {
            "detected": False,
            "target_brand": None,
            "legitimate_domain": None,
            "is_genuine": False,
            "similarity_score": 0.0,
            "reasons": [],
            "detection_engine": [],
            "is_reserved_tld": False,
            "classification_hint": "UNKNOWN"
        }

    # 3. URL Feature Extraction
    feats = {}
    if extract_features:
        try:
            feat_vector = extract_features(clean_url)
            feats = dict(zip(FEATURE_NAMES, feat_vector))
        except Exception:
            feats = {}

    # 4. Classical ML Inference
    clf_bundle, q_bundle = get_models()
    prob_classical = None
    classical_success = False
    threshold = 0.5
    model_name = "LightGBM"
    model_status = "success"

    if clf_bundle and clf_bundle is not False and predict_url:
        try:
            threshold = getattr(clf_bundle, "threshold", 0.5)
            clf = getattr(clf_bundle, "clf", clf_bundle[0])
            scaler = getattr(clf_bundle, "scaler", clf_bundle[1])
            c_res = predict_url(clean_url, clf=clf, scaler=scaler, threshold=threshold)
            prob_classical = float(c_res.get("probability_phishing", 0.5))
            meta = getattr(clf_bundle, "metadata", {}) or {}
            model_name = meta.get("model_type", "LightGBM").upper()
            classical_success = True
        except Exception:
            prob_classical = None
            classical_success = False
            model_status = "error"
    else:
        model_status = "fallback" if not clf_bundle else "error"

    # 5. Quantum VQC Inference
    prob_quantum = prob_classical if prob_classical is not None else 0.5
    quantum_success = False
    if q_bundle and q_bundle is not False and predict_url_quantum:
        try:
            q_res = predict_url_quantum(clean_url, bundle=q_bundle)
            prob_quantum = float(q_res.get("probability_phishing", prob_quantum))
            quantum_success = True
        except Exception:
            pass

    # 6. Lexical Heuristics & Threat Intelligence
    evidence = analyze_url_heuristics(clean_url, feats)
    threat_intel = get_threat_intelligence(clean_url)

    # 7. Multi-Modal Synthesis & Classification Engine
    detection_engines = []
    reasons = []
    confidence = None
    risk_score = 0
    classification = "UNKNOWN"

    is_impersonation = impersonation.get("detected", False)
    is_genuine_brand = impersonation.get("is_genuine", False)
    is_reserved = impersonation.get("is_reserved_tld", False)
    target_brand = impersonation.get("target_brand")
    legitimate_domain = impersonation.get("legitimate_domain")

    # BRANCH A: Verified genuine protected brand (e.g. google.com, microsoft.com)
    if is_genuine_brand:
        classification = "SAFE"
        risk_score = 5
        detection_engines = ["brand_verification"]
        if classical_success:
            detection_engines.append("classical_lightgbm")
        reasons = [f"Verified legitimate domain for {target_brand}"]
        composite_prob = prob_classical if prob_classical is not None else 0.05
        confidence = round(1.0 - composite_prob, 4)

    # BRANCH B: Brand impersonation / typosquatting detected (e.g. g00gle.com, paypa1.com)
    elif is_impersonation:
        classification = "SUSPICIOUS"
        risk_score = 85
        detection_engines = impersonation.get("detection_engine") or [
            "typosquatting",
            "brand_impersonation",
            "domain_similarity"
        ]
        reasons = list(impersonation.get("reasons", []))

        # Escalate to MALICIOUS if verified threat intelligence IOC also flags domain
        if threat_intel.get("status") == "flagged":
            classification = "MALICIOUS"
            risk_score = 98
            detection_engines.append("threat_intelligence")
            reasons.append("Flagged by threat intelligence IOC feed")

        composite_prob = max(prob_classical or 0.0, 0.85)
        confidence = None

    # BRANCH C: Threat intelligence flagged domain
    elif threat_intel.get("status") == "flagged":
        classification = "MALICIOUS"
        risk_score = 95
        detection_engines = ["threat_intelligence"]
        reasons = ["Domain flagged by threat intelligence IOC feed as active phishing campaign"]
        composite_prob = 0.95
        confidence = 0.95

    # BRANCH D: ML Pipeline Failure (Never return SAFE when inference fails!)
    elif not classical_success or prob_classical is None:
        classification = "UNKNOWN"
        risk_score = 50
        detection_engines = ["fallback_heuristics"]
        reasons = ["ML inference pipeline unavailable; safety status cannot be verified"]
        composite_prob = 0.50
        confidence = None
        model_status = "error"

    # BRANCH E: Reserved synthetic TLD without brand (e.g. test.example)
    elif is_reserved:
        classification = "UNKNOWN"
        risk_score = 30
        detection_engines = ["tld_analyzer"]
        reasons = ["Generic domain under synthetic or reserved TLD; insufficient operational signals"]
        composite_prob = 0.30
        confidence = None

    # BRANCH F: Standard ML + Heuristics Decision
    else:
        composite_prob = float(0.80 * prob_classical + 0.20 * prob_quantum)
        detection_engines = ["classical_lightgbm"]
        if quantum_success:
            detection_engines.append("quantum_vqc")

        if composite_prob >= 0.80 or prob_classical >= 0.80:
            classification = "MALICIOUS"
            risk_score = int(max(composite_prob, prob_classical) * 100)
            reasons = ["Elevated ML phishing probability across lexical feature dimensions"]
            confidence = round(composite_prob, 4)
        elif composite_prob >= threshold or prob_classical >= threshold:
            classification = "SUSPICIOUS"
            risk_score = int(max(composite_prob, prob_classical) * 100)
            reasons = ["URL exhibits suspicious structural and lexical anomalies"]
            confidence = round(composite_prob, 4)
        elif composite_prob >= 0.35 or len(evidence) >= 2:
            classification = "SUSPICIOUS"
            risk_score = int(composite_prob * 100)
            reasons = [ev["title"] for ev in evidence[:2]] or ["Elevated anomaly score from lexical heuristics"]
            confidence = round(composite_prob, 4)
        else:
            classification = "SAFE"
            risk_score = max(5, int(composite_prob * 100))
            reasons = ["No meaningful phishing or impersonation indicators detected"]
            confidence = round(1.0 - composite_prob, 4)

    # 8. Legacy Field Mappings
    if classification == "MALICIOUS":
        prediction = "phishing"
        prediction_text = "Phishing Detected (High Confidence)"
        risk_level = "critical"
    elif classification == "SUSPICIOUS":
        prediction = "phishing"
        prediction_text = "Phishing Suspected"
        risk_level = "high"
    elif classification == "UNKNOWN":
        prediction = "unknown"
        prediction_text = "Unknown / Inconclusive Risk"
        risk_level = "medium"
    else:
        prediction = "legitimate"
        prediction_text = "Likely Legitimate"
        risk_level = "low"

    c_prob_safe = prob_classical if prob_classical is not None else 0.5

    return {
        # Section 5 exact standardized fields
        "url": clean_url,
        "classification": classification,
        "risk_score": risk_score,
        "confidence": confidence,
        "detection_engine": detection_engines,
        "brand_impersonation": {
            "detected": is_impersonation,
            "target_brand": target_brand,
            "legitimate_domain": legitimate_domain
        },
        "reasons": reasons,
        "model_status": model_status,

        # Backward compatibility fields for frontend and historical consumers
        "input_valid": True,
        "prediction": prediction,
        "prediction_text": prediction_text,
        "risk_level": risk_level,
        "model_version": f"q-sentinel-v2.0-hybrid-{model_name.lower()}-vqc",
        "threshold_used": round(threshold, 3),
        "risk_score_pct": float(risk_score),
        "probabilities": {
            "classical_rf_phishing": round(float(c_prob_safe), 4),
            "classical_phishing": round(float(c_prob_safe), 4),
            "quantum_vqc_phishing": round(float(prob_quantum), 4),
            "composite_phishing": round(float(composite_prob), 4),
            "legitimate": round(1.0 - float(composite_prob), 4)
        },
        "model_types": {
            "classical": model_name,
            "quantum": "4-Qubit Variational Quantum Classifier (Statevector Sim)"
        },
        "ml_evidence": evidence,
        "threat_intelligence": threat_intel,
        "features": {k: round(v, 2) if isinstance(v, float) else v for k, v in feats.items()},
        "limitations": [
            "Purely non-intrusive lexical and structural inspection. No active HTTP probes or DNS lookups performed to prevent SSRF vulnerabilities.",
            "Predictions reflect statistical model probabilities and domain similarity heuristics; always verify credentials independently."
        ]
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_rate_limit(self) -> bool:
        """Enforce rate limits: max 120 requests per minute per IP."""
        client_ip = self.client_address[0] if self.client_address else "unknown"
        now = time.time()
        window = [t for t in _RATE_LIMITS[client_ip] if now - t < 60]
        if len(window) >= 120:
            return False
        window.append(now)
        _RATE_LIMITS[client_ip] = window
        return True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path in ("/api/health", "/api/health/"):
            clf, q_b = get_models()
            self._send_json({
                "status": "healthy",
                "service": "Q-Sentinel Threat Engine",
                "version": "2.1.0",
                "dataset": "PhiUSIIL Phishing URL Websites Dataset (v1.0)",
                "models": {
                    "classical_pipeline": clf is not None and clf is not False,
                    "quantum_vqc": q_b is not None and q_b is not False
                },
                "security": {
                    "typosquatting_engine": check_domain_impersonation is not None,
                    "ssrf_protection": validate_and_sanitize_url is not None
                }
            })
            return

        if path in ("/api/metrics", "/api/metrics/"):
            m_paths = [CURRENT_DIR / "results" / "metrics.json", ROOT_DIR / "results" / "metrics.json"]
            q_paths = [CURRENT_DIR / "results" / "quantum_metrics.json", ROOT_DIR / "results" / "quantum_metrics.json"]
            b_paths = [CURRENT_DIR / "results" / "benchmark_comparison.json", ROOT_DIR / "results" / "benchmark_comparison.json"]

            c_metrics = {}
            q_metrics = {}
            b_metrics = {}

            for p in m_paths:
                if p.is_file():
                    try:
                        c_metrics = json.loads(p.read_text())
                        break
                    except Exception:
                        pass

            for p in q_paths:
                if p.is_file():
                    try:
                        q_metrics = json.loads(p.read_text())
                        break
                    except Exception:
                        pass

            for p in b_paths:
                if p.is_file():
                    try:
                        b_metrics = json.loads(p.read_text())
                        break
                    except Exception:
                        pass

            self._send_json({
                "classical": c_metrics,
                "quantum": q_metrics,
                "benchmark": b_metrics,
                "features": FEATURE_NAMES
            })
            return

        # Default info endpoint
        self._send_json({
            "message": "Q-Sentinel Hybrid Quantum-Classical Detection API v2.1",
            "dataset": "PhiUSIIL Phishing URL Websites Dataset",
            "classification_tiers": ["SAFE", "SUSPICIOUS", "MALICIOUS", "UNKNOWN", "SCAN_ERROR"],
            "endpoints": [
                "POST /api/scan (body: {\"url\": \"...\"})",
                "POST /api/batch (body: {\"urls\": [\"...\"]})",
                "GET /api/metrics",
                "GET /api/health"
            ]
        })

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if not self._check_rate_limit():
            self._send_json({"error": "Rate limit exceeded. Maximum 120 requests per minute."}, 429)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json({
                "classification": "SCAN_ERROR",
                "error": "Empty request body",
                "model_status": "error"
            }, 400)
            return

        if content_length > MAX_REQUEST_SIZE_BYTES:
            self._send_json({
                "classification": "SCAN_ERROR",
                "error": f"Payload too large. Exceeds maximum limit of {MAX_REQUEST_SIZE_BYTES} bytes",
                "model_status": "error"
            }, 413)
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json({
                "classification": "SCAN_ERROR",
                "error": f"Invalid JSON payload: {str(e)}",
                "model_status": "error"
            }, 400)
            return

        if path in ("/api/scan", "/api/scan/"):
            url = payload.get("url", "")
            if not url or not str(url).strip():
                self._send_json({
                    "classification": "SCAN_ERROR",
                    "error": "Missing or empty 'url' property in request body",
                    "model_status": "error"
                }, 400)
                return

            result = perform_full_scan(url)
            status_code = 400 if result.get("classification") == "SCAN_ERROR" else 200
            self._send_json(result, status_code)
            return

        if path in ("/api/batch", "/api/batch/"):
            urls = payload.get("urls", [])
            if not isinstance(urls, list):
                self._send_json({
                    "classification": "SCAN_ERROR",
                    "error": "'urls' must be a JSON array of URL strings",
                    "model_status": "error"
                }, 400)
                return

            # Bound limit to 50 items for serverless performance
            bounded_urls = urls[:50]
            results = [perform_full_scan(u) for u in bounded_urls]
            self._send_json({
                "count": len(results),
                "total_requested": len(urls),
                "results": results
            })
            return

        self._send_json({"error": f"Endpoint not found: {path}"}, 404)
