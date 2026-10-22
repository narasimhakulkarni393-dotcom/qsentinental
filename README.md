# 🛡️ Q-Sentinel

**Hybrid Quantum–Classical AI for Phishing URL Detection**

> Version 2.0 — Classical baseline + variational quantum classifier

Q-Sentinel analyses URL strings with machine learning to predict whether a URL
resembles a phishing attempt.  v1 is a classical RandomForest pipeline; v2 adds
a hybrid quantum classifier: a RandomForest ranks the lexical features, and a
variational quantum circuit (one qubit per selected feature) makes the decision.
The quantum circuit is simulated in pure NumPy — no quantum SDK or hardware needed.

---

## Quick Start (Windows PowerShell)

```powershell
cd q-sentinel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest                                   # run tests

# classical baseline
python train.py --data data/phishing_site_urls.csv --url-col URL --label-col Label

# quantum model (same train/test split -> directly comparable)
python train_quantum.py --data data/phishing_site_urls.csv --url-col URL --label-col Label

streamlit run app.py                     # dashboard (+ ⚛️ Quantum Lab page in the sidebar)
```

Useful quantum options: `--qubits 4` (features used), `--layers 3`, `--steps 120`,
`--max-train 800` (samples the circuit trains on; simulation is slower than a forest).

## How the quantum part works

1. **Classical step** – a RandomForest ranks the 13 URL features; the top *n* are
   log-scaled and mapped to angles in [0, π].
2. **Encoding** – feature *i* is loaded onto qubit *i* with an `RY(x_i)` rotation.
3. **Variational circuit** – *L* layers of trainable `RY`/`RZ` rotations plus a ring of
   CNOTs for entanglement.
4. **Readout** – `<Z>` on qubit 0 gives `P(phishing) = (1 + <Z>) / 2`.
5. **Training** – parameter-shift gradients (exact, hardware-compatible), Adam optimiser,
   binary cross-entropy loss.

## Project Structure

```
q-sentinel/
├── app.py                 # Streamlit dashboard
├── pages/1_Quantum_Lab.py # Quantum vs classical comparison page
├── train.py               # Classical training CLI
├── train_quantum.py       # Quantum training CLI
├── pytest.ini
├── requirements.txt
├── src/
│   ├── features.py        # URL feature extraction (13 features)
│   ├── data_loader.py     # CSV loading & validation
│   ├── model.py           # Classical training / evaluation / prediction
│   ├── quantum.py         # NumPy state-vector simulator + VQC
│   └── quantum_model.py   # Hybrid pipeline, persistence, prediction
├── tests/                 # features, data loader, model, quantum
├── data/                  # README + 20-row smoke-test sample
├── models/                # Saved artefacts (git-ignored)
└── results/               # Metrics & predictions (git-ignored)
```

## Features Extracted

`url_length`, `hostname_length`, `num_dots`, `num_subdomains`, `num_hyphens`,
`num_digits`, `num_special_chars`, `num_at_symbols`, `is_ip_address`, `is_https`,
`num_path_separators`, `num_query_params`, `has_unusual_port`.

## Importing a Real Dataset

See [`data/README.md`](data/README.md).  **Label convention:** `1 = phishing`, `0 = legitimate`.

## ⚠️ Disclaimer

This tool provides statistical estimates based on URL lexical patterns.  It is
**not** a substitute for professional cybersecurity tools.  Never rely solely on
this prediction to determine URL safety.  The quantum model is a research
prototype and is not expected to beat the classical baseline.

## Licence

MIT
