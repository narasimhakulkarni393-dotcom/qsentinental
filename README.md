# 🛡️ Q-Sentinel

**Hybrid Quantum–Classical AI for Phishing URL Detection**

> Version 1.0 — Classical Baseline

Q-Sentinel analyses URL strings using machine learning to predict whether a
URL resembles a phishing attempt.  This first version implements a classical
RandomForest pipeline; a quantum extension is planned for v2.

---

## Quick Start (Windows PowerShell)

```powershell
# 1. Navigate to the project
cd q-sentinel

# 2. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest tests/ -v

# 5. Train the model (you need a real dataset — see data/README.md)
python train.py --data data/phishing_site_urls.csv --url-col URL --label-col Label

# 6. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
q-sentinel/
├── app.py               # Streamlit dashboard
├── train.py             # Training CLI script
├── requirements.txt     # Python dependencies
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── features.py      # URL feature extraction (13 features)
│   ├── data_loader.py   # CSV loading & validation
│   └── model.py         # Training, evaluation, prediction
├── tests/
│   ├── test_features.py
│   ├── test_data_loader.py
│   └── test_model.py
├── data/
│   ├── README.md        # Instructions for importing real datasets
│   └── sample_urls.csv  # 20-row smoke-test data (NOT for real training)
├── models/              # Saved model artefacts (git-ignored)
└── results/             # Evaluation metrics & predictions (git-ignored)
```

## Features Extracted

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Total characters in the URL |
| 2 | `hostname_length` | Characters in the hostname |
| 3 | `num_dots` | Number of dots in the full URL |
| 4 | `num_subdomains` | Hostname parts minus domain + TLD |
| 5 | `num_hyphens` | Hyphen count |
| 6 | `num_digits` | Digit count |
| 7 | `num_special_chars` | Non-alphanumeric, non-dot, non-hyphen chars |
| 8 | `num_at_symbols` | `@` count |
| 9 | `is_ip_address` | Hostname is an IPv4/IPv6 address |
| 10 | `is_https` | Scheme is HTTPS |
| 11 | `num_path_separators` | `/` in the path (excluding leading) |
| 12 | `num_query_params` | Number of query-string parameters |
| 13 | `has_unusual_port` | Port is not 80 / 443 / 8080 / 8443 |

## Importing a Real Dataset

See [`data/README.md`](data/README.md) for recommended public datasets and
import instructions.

**Label convention:** `1 = phishing`, `0 = legitimate`.

## ⚠️ Disclaimer

This tool provides statistical estimates based on URL lexical patterns.  It is
**not** a substitute for professional cybersecurity tools.  Never rely solely
on this prediction to determine URL safety.

## Licence

MIT
