"""Dataset loading, validation, and domain-disjoint splitting for Q-Sentinel.

Supports:
- Generic phishing URL CSVs (with configurable URL and Label columns).
- Automatic detection and mapping for the PhiUSIIL Phishing URL Websites Dataset.
- Extraction of domain groups for domain-disjoint evaluation to prevent model memorization.
- Label convention throughout Q-Sentinel:
    1 = Phishing
    0 = Legitimate
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# Recognised text -> numeric label mappings (case-insensitive).
_LABEL_MAP = {
    "phishing": 1,
    "malicious": 1,
    "bad": 1,
    "spam": 1,
    "legitimate": 0,
    "benign": 0,
    "good": 0,
    "safe": 0,
}


def _extract_domain(url: str) -> str:
    """Safely extract registered hostname/domain for grouping splits."""
    if not isinstance(url, str) or not url.strip():
        return "unknown"
    clean = url.strip()
    if "://" not in clean:
        clean = "http://" + clean
    try:
        parsed = urlparse(clean)
        return (parsed.hostname or "unknown").lower()
    except Exception:
        return "unknown"


def load_dataset(
    csv_path: str | Path,
    url_col: str = "url",
    label_col: str = "label",
    is_phiusiil: bool | None = None,
    nrows: int | None = None,
    include_domain: bool = False,
) -> pd.DataFrame:
    """Load and validate a labelled URL dataset from *csv_path*.

    Returns a DataFrame with columns: ``url`` (str), ``label`` (int, 0 or 1),
    and optionally ``domain`` (str, if include_domain=True).

    Label convention:
        1 = Phishing
        0 = Legitimate

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file.
    url_col : str
        Column name containing the URL string (case-insensitive lookup fallback).
    label_col : str
        Column name containing the target label.
    is_phiusiil : bool or None
        If True, or if PhiUSIIL signature columns (e.g., TLDLegitimateProb) are detected,
        labels are inverted: PhiUSIIL's 0 (phishing) becomes 1, and 1 (legitimate) becomes 0.
    nrows : int or None
        Optional maximum number of rows to load.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Read preview to inspect column names
    preview = pd.read_csv(csv_path, nrows=5)
    cols = list(preview.columns)

    # Auto-detect column case if exact match not found
    if url_col not in cols:
        lower_cols = {c.lower(): c for c in cols}
        if url_col.lower() in lower_cols:
            url_col = lower_cols[url_col.lower()]

    if label_col not in cols:
        lower_cols = {c.lower(): c for c in cols}
        if label_col.lower() in lower_cols:
            label_col = lower_cols[label_col.lower()]

    if url_col not in cols:
        raise ValueError(
            f"URL column '{url_col}' not found. Available columns: {cols[:10]}..."
        )
    if label_col not in cols:
        raise ValueError(
            f"Label column '{label_col}' not found. Available columns: {cols[:10]}..."
        )

    # Detect if dataset is PhiUSIIL
    phiusiil_markers = {"URLLength", "TLDLegitimateProb", "DomainTitleMatchScore", "IsDomainIP"}
    detected_phiusiil = phiusiil_markers.issubset(set(cols))
    if is_phiusiil is None:
        is_phiusiil = detected_phiusiil

    df = pd.read_csv(csv_path, usecols=[url_col, label_col], nrows=nrows)
    df = df[[url_col, label_col]].copy()
    df.columns = ["url", "label"]

    # Drop missing values
    before = len(df)
    df.dropna(subset=["url", "label"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing URL or label.")

    if df.empty:
        raise ValueError("Dataset is empty after removing missing values.")

    # Drop duplicate URLs
    before_dupes = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    if before_dupes - len(df) > 0:
        print(f"  Removed {before_dupes - len(df)} duplicate URLs.")

    # Map labels to 0/1
    if pd.api.types.is_string_dtype(df["label"]):
        df["label"] = df["label"].str.strip().str.lower()
        mapped = df["label"].map(_LABEL_MAP)
        unmapped = df["label"][mapped.isna()].unique()
        if len(unmapped) > 0:
            raise ValueError(
                f"Cannot map these labels to 0/1: {list(unmapped)}. "
                f"Accepted text labels: {list(_LABEL_MAP.keys())}"
            )
        df["label"] = mapped.astype(int)
    else:
        unique = set(df["label"].unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Numeric labels must be 0 or 1, but found: {unique}"
            )
        df["label"] = df["label"].astype(int)

    # Handle PhiUSIIL label inversion (0 = phishing, 1 = legitimate in raw dataset)
    if is_phiusiil:
        print("  [PhiUSIIL Detected] Inverting labels: raw 0 (phishing) -> 1, raw 1 (legitimate) -> 0.")
        df["label"] = 1 - df["label"]

    # Ensure both classes are present
    classes = set(df["label"].unique())
    if classes != {0, 1}:
        raise ValueError(
            f"Dataset must contain both classes (0 and 1). Found: {classes}"
        )

    # Extract domain if requested
    if include_domain:
        df["domain"] = df["url"].apply(_extract_domain)
    else:
        df = df[["url", "label"]]

    return df.reset_index(drop=True)
