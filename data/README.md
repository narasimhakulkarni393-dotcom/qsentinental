# Data Directory

## sample_urls.csv

**This file is illustrative test data only.**  It contains 20 hand-crafted
example URLs (10 labelled "legitimate", 10 labelled "phishing") and exists
solely to support automated smoke tests.  

**Do NOT use it to train a real model or report detection performance.**

## Using a Real Dataset

To train a meaningful phishing-detection model you need a large, labelled
dataset.  Recommended public sources:

| Dataset | URL | Size | Format |
|---------|-----|------|--------|
| Kaggle – Phishing Site URLs | https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls | ~550 k | CSV with `URL`, `Label` columns |
| PhiUSILL | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset | ~235 k | CSV |

### Import steps

1. Download the CSV and place it in this `data/` folder.
2. Open the file and identify the URL column and the label column.
3. Run training with the correct column names, for example:

```bash
python train.py --data data/phishing_site_urls.csv --url-col URL --label-col Label
```

The loader automatically maps common text labels ("phishing" / "legitimate",
"bad" / "good", etc.) to 1 / 0.  Label 1 = phishing, label 0 = legitimate.
