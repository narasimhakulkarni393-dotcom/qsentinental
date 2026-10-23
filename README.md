# 🛡️ Q-Sentinel: Hybrid Quantum-Classical Phishing URL Detection Engine

[![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://qsentinental.vercel.app)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg?style=for-the-badge)](LICENSE)

**Live Production Web App:** [https://qsentinental.vercel.app](https://qsentinental.vercel.app)

---

## 🌐 Overview

**Q-Sentinel** is an advanced AI cybersecurity system designed to identify and classify deceptive phishing URLs without performing dangerous external network queries or DNS lookups (zero-network inspection prevents Server-Side Request Forgery and DNS-rebinding risks).

The platform introduces a **Dual-Engine Architecture**:
1. 🌲 **Classical Random Forest Classifier:** Evaluates 13 lexical and structural URL indicators.
2. ⚛️ **Variational Quantum Classifier (VQC):** Maps non-linear feature distributions onto a multi-qubit parameterized quantum state space using Pauli rotation encodings ($R_y, R_z$) and entanglement layers.

---

## 🚀 Key Features

* **Real-Time Cyber Threat Dashboard:** Dark-mode glassmorphic interface with interactive threat gauge and live scanning telemetry.
* **Dual-Engine Verdict:** Side-by-side risk score from Classical Random Forest and Quantum VQC simulation.
* **Explainable ML Evidence:** Human-readable explanations for security indicators (e.g., IP-address hostnames, non-standard ports, multi-level subdomain masquerading, lure keywords).
* **Batch Threat Inspector:** Bulk scanning (up to 50 URLs concurrently) with instant CSV export.
* **Quantum Lab:** Interactive parameterized circuit schematic, benchmark comparisons, and BCE optimizer loss convergence history.
* **Serverless Edge API:** Scalable REST endpoints on Vercel (`/api/scan`, `/api/batch`, `/api/metrics`, `/api/health`).

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
│   ├── features.py            # 13-feature lexical extractor
│   ├── data_loader.py         # CSV loading, sanitization & validation
│   ├── model.py               # Classical Random Forest pipeline
│   ├── quantum.py             # Pure NumPy state-vector quantum simulator
│   └── quantum_model.py       # Angle scaler & hybrid VQC training pipeline
├── models/                    # Serialized model artifacts
│   ├── model.joblib           # Trained Classical Random Forest + Scaler
│   ├── quantum_model.joblib   # Trained VQC weights + angle scaler
│   └── feature_names.json     # Feature schema
├── results/                   # Evaluation artifacts
│   ├── metrics.json           # Precision, Recall, F1, FPR, Confusion Matrix
│   └── quantum_metrics.json   # Quantum circuit training history & loss
├── tests/                     # Unit test suite
├── app.py                     # Local Streamlit Dashboard
├── pages/1_Quantum_Lab.py     # Local Streamlit Quantum comparison page
├── train.py                   # Classical training CLI
├── train_quantum.py           # Quantum VQC training CLI
├── vercel.json                # Vercel deployment configuration
├── requirements.txt           # Production serverless dependencies
└── requirements-dev.txt       # Local development & Streamlit dependencies
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
python -m pytest -q
```

### 3. Launch Local Streamlit UI
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
  "url": "http://appleid.apple.com.secure-login.info/verify"
}
```
**Response:**
```json
{
  "input_valid": true,
  "url": "http://appleid.apple.com.secure-login.info/verify",
  "prediction": "phishing",
  "prediction_text": "Phishing Detected",
  "risk_level": "critical",
  "risk_score_pct": 96.0,
  "probabilities": {
    "classical_rf_phishing": 0.92,
    "quantum_vqc_phishing": 0.9037,
    "composite_phishing": 0.96,
    "legitimate": 0.04
  },
  "ml_evidence": [
    {
      "feature": "num_subdomains",
      "severity": "warning",
      "title": "Deep Subdomain Hierarchy",
      "description": "URL exhibits 3 subdomain levels with 4 dots, a common domain masquerading technique."
    }
  ]
}
```

### 2. Service Health
**Endpoint:** `GET /api/health`
```json
{
  "status": "healthy",
  "service": "Q-Sentinel Threat Engine",
  "version": "1.2.0",
  "models": {
    "classical_rf": true,
    "quantum_vqc": true
  }
}
```

---

## ⚠️ Responsible AI & Security Disclaimer

Q-Sentinel provides statistical estimations based on lexical patterns and simulated quantum state rotations. It is **not** a substitute for enterprise-grade endpoint security, authenticated identity assertions, or multi-factor authentication (MFA).

## 📄 License

Distributed under the [MIT License](LICENSE).
