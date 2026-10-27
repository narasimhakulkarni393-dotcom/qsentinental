# 🛡️ Q-Sentinel: Hybrid Quantum-Classical Phishing URL Detection Engine

[![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://qsentinental.vercel.app)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg?style=for-the-badge)](LICENSE)

**Live Production Web App:** [https://qsentinental.vercel.app](https://qsentinental.vercel.app)

---

## 🌐 Overview

**Q-Sentinel** is an enterprise-grade AI cybersecurity intelligence platform engineered to detect and classify deceptive phishing URLs without conducting active HTTP probes or resolving DNS. Non-intrusive lexical inspection prevents Server-Side Request Forgery (SSRF) and DNS-rebinding risks while delivering ultra-low latency (< 0.1 ms per evaluation).

The platform introduces a **Dual-Engine Architecture**:
1. ⚡ **Production Classical Engine (LightGBM):** Evaluates 25 high-discrimination lexical and structural indicators, tuned via validation threshold calibration to minimize false negatives and achieve **99.17% phishing recall**.
2. ⚛️ **Variational Quantum Classifier (VQC):** Parameterized quantum circuit simulator with Pauli rotation encodings ($R_y, R_z$) and entanglement layers, mapping non-linear feature interactions onto a 4-qubit Hilbert state space.

---

## 📊 Benchmark & Evaluation Results

Evaluated on 40,000 URLs from the **PhiUSIIL Phishing URL Websites Dataset** on held-out test data across both **Domain-Disjoint** (unseen domains) and **Random Stratified** splits:

| Split Strategy | Model Architecture | Phishing Recall | Precision | F1-Score | FPR | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Domain-Disjoint** | **LightGBM (Production)** | **99.17%** | **99.97%** | **0.9957** | **0.02%** | **0.01 ms** |
| Domain-Disjoint | Random Forest | 99.14% | 100.00% | 0.9957 | 0.00% | 0.08 ms |
| Domain-Disjoint | Logistic Regression | 98.60% | 99.97% | 0.9928 | 0.02% | 0.005 ms |
| Random Stratified | LightGBM | 98.99% | 99.94% | 0.9946 | 0.04% | 0.01 ms |
| Random Stratified | Random Forest | 99.02% | 99.91% | 0.9946 | 0.06% | 0.08 ms |
| Random Stratified | Logistic Regression | 98.96% | 99.88% | 0.9942 | 0.08% | 0.005 ms |
| Random Stratified | VQC (4-Qubit Simulator) | 76.72% | 99.97% | 0.8681 | 0.01% | 0.42 ms |

### Held-Out Test Confusion Matrix (Domain-Disjoint: 7,923 Unseen Domain URLs)
* **True Negatives (Legitimate Correctly Passed):** 4,789
* **False Positives (Legitimate Flagged):** 1
* **False Negatives (Missed Phishing):** 26 (down from hundreds in uncalibrated baselines)
* **True Positives (Phishing Correctly Caught):** 3,107
* **Calibrated Decision Threshold ($\tau$):** `0.830`

---

## 🔍 Preprocessing & Anti-Leakage Pipeline

### 1. PhiUSIIL Dataset Integration
* Automatically detects signature columns of the PhiUSIIL dataset (`kaggleprollc/phishing-url-websites-dataset-phiusiil`).
* **Critical Label Inversion:** In the raw PhiUSIIL dataset, label `0` indicates phishing and label `1` indicates legitimate. Q-Sentinel transparently inverts labels to enforce standard cybersecurity convention (`1 = Phishing`, `0 = Legitimate`).
* **Zero Feature Leakage:** Drops precomputed HTML/DOM columns (`LineOfCode`, `NoOfiFrame`, `HasTitle`, `Bank`, `Pay`, `Crypto`, `HasPasswordField`) which cannot be observed by a real-time URL scanner without browsing external web pages.

### 2. URL Invariant Normalization
* Normalizes `www.` prefixes and URL schemes so apex domains (`google.com`) and prefixed domains (`www.google.com`) produce identical, unbiased feature vectors.
* Safely processes IPv4, IPv6, Punycode/IDN homographs, non-standard ports, and URL-encoded payloads.

### 3. Extracted Lexical Features (25 Features)
1. `url_length`
2. `hostname_length`
3. `path_length`
4. `query_length`
5. `num_dots`
6. `num_subdomains`
7. `num_hyphens`
8. `hyphen_in_domain`
9. `num_digits`
10. `digit_ratio_url`
11. `digit_ratio_domain`
12. `num_special_chars`
13. `num_at_symbols`
14. `is_ip_address`
15. `is_https`
16. `has_unusual_port`
17. `has_punycode`
18. `num_path_separators`
19. `num_query_params`
20. `has_double_slash_path`
21. `entropy_url` (Shannon entropy)
22. `entropy_domain`
23. `has_suspicious_tld` (.tk, .top, .xyz, .cf, etc.)
24. `num_sensitive_keywords` (login, verify, banking, oauth, etc.)
25. `has_brand_in_subdomain` (brand spoofing detection)

---

## 📁 Repository Structure

```
qsentinental/
├── api/                       # Vercel serverless REST API
│   ├── index.py               # Main API handler & routing
│   ├── scan.py                # POST /api/scan endpoint
│   ├── batch.py               # POST /api/batch endpoint
│   ├── metrics.py             # GET /api/metrics endpoint
│   └── health.py              # GET /api/health endpoint
├── public/                    # Frontend Web Application
│   └── index.html             # Cyberpunk/SOC dark-mode responsive dashboard
├── src/                       # Core ML and Quantum engine
│   ├── features.py            # 25-feature lexical extractor & invariant normalizer
│   ├── data_loader.py         # CSV loading, PhiUSIIL auto-detect & domain-disjoint splitting
│   ├── model.py               # LightGBM/RF pipeline & threshold calibration
│   ├── quantum.py             # Pure NumPy state-vector quantum simulator
│   └── quantum_model.py       # Angle scaler & hybrid VQC training pipeline
├── models/                    # Serialized model artifacts
│   ├── model.joblib           # Trained LightGBM bundle (model, scaler, calibrated threshold)
│   ├── quantum_model.joblib   # Trained VQC weights + angle scaler
│   └── feature_names.json     # Feature schema
├── results/                   # Evaluation artifacts
│   ├── metrics.json           # Comprehensive test metrics
│   ├── benchmark_comparison.json # Multi-model cross-split benchmark results
│   └── test_predictions.csv   # Predictions on held-out test split
├── tests/                     # 42 unit and integration tests
├── app.py                     # Local Streamlit Dashboard
├── train.py                   # Classical training & multi-model benchmark CLI
├── train_quantum.py           # Quantum VQC training CLI
├── vercel.json                # Vercel deployment configuration
├── requirements.txt           # Lean serverless dependencies
└── requirements-dev.txt       # Local development & testing dependencies
```

---

## ⚡ Quick Start (Local Setup)

### 1. Clone & Activate Environment
```powershell
git clone https://github.com/narasimhakulkarni393-dotcom/qsentinental.git
cd qsentinental

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Run Tests
```powershell
python -m pytest -v
```
All 42 test cases covering features, data loader, models, and quantum simulator will execute.

### 3. Run Benchmark / Training
```powershell
# Train production LightGBM model with domain-disjoint split:
python train.py --split-mode domain_disjoint

# Run comprehensive comparative benchmark:
python train.py --benchmark
```

### 4. Launch Local Streamlit Dashboard
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ REST API Documentation

### 1. Scan a URL
**Endpoint:** `POST /api/scan`  
**Request:**
```json
{
  "url": "http://paypa1-verify.com/login?token=abc"
}
```
**Response:**
```json
{
  "input_valid": true,
  "url": "http://paypa1-verify.com/login?token=abc",
  "prediction": "phishing",
  "prediction_text": "Phishing Detected (High Confidence)",
  "risk_level": "critical",
  "threshold_used": 0.83,
  "risk_score_pct": 98.0,
  "probabilities": {
    "classical_phishing": 0.9999,
    "quantum_vqc_phishing": 0.8443,
    "composite_phishing": 0.98,
    "legitimate": 0.02
  },
  "ml_evidence": [
    {
      "feature": "suspicious_keywords",
      "severity": "warning",
      "title": "Targeted Authentication Keywords",
      "description": "Contains security-sensitive lure keywords: login, verify, token"
    }
  ]
}
```

### 2. Batch Scan
**Endpoint:** `POST /api/batch`  
**Request:**
```json
{
  "urls": [
    "https://google.com",
    "http://paypa1-verify.com/signin",
    "https://github.com"
  ]
}
```

### 3. Metrics
**Endpoint:** `GET /api/metrics`  
Returns current model metrics, confusion matrix, and benchmark comparison data.

### 4. Health
**Endpoint:** `GET /api/health`  
Returns runtime health status and active loaded engines.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
