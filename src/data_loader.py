"""Dataset loading and validation for Q-Sentinel.

Expected CSV schema
-------------------
- A column containing raw URL strings.
- A column containing binary labels (phishing = 1, legitimate = 0).

The column names are configurable.  Common text labels such as
"phishing" / "legitimate" or "bad" / "good" are automatically mapped to 1 / 0.
"""

from pathlib import Path

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


def load_dataset(
    csv_path: str | Path,
    url_col: str = "url",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load and validate a labelled URL dataset from *csv_path*.

    Returns a DataFrame with exactly two columns: ``url`` (str) and
    ``label`` (int, 0 or 1).

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing, labels cannot be mapped, or the
        dataset is otherwise unusable.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Validate columns ------------------------------------------------
    if url_col not in df.columns:
        raise ValueError(
            f"URL column '{url_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[url_col, label_col]].copy()
    df.columns = ["url", "label"]

    # --- Drop rows with missing values -----------------------------------
    before = len(df)
    df.dropna(subset=["url", "label"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing URL or label.")

    if df.empty:
        raise ValueError("Dataset is empty after removing missing values.")

    # --- Map text labels to 0 / 1 ----------------------------------------
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
        # Numeric — ensure only 0 and 1 are present.
        unique = set(df["label"].unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Numeric labels must be 0 or 1, but found: {unique}"
            )
        df["label"] = df["label"].astype(int)

    # --- Require both classes --------------------------------------------
    classes = set(df["label"].unique())
    if classes != {0, 1}:
        raise ValueError(
            f"Dataset must contain both classes (0 and 1). Found: {classes}"
        )

    return df.reset_index(drop=True)
