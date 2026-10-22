"""Q-Sentinel — Model training CLI.

Usage
-----
    python train.py --data data/phishing_site_urls.csv
    python train.py --data data/dataset.csv --url-col URL --label-col Label
"""

import argparse
import sys

from src.data_loader import load_dataset
from src.model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Q-Sentinel phishing-URL classifier.",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to a labelled CSV file (must have a URL column and a label column).",
    )
    parser.add_argument(
        "--url-col", default="url",
        help="Name of the CSV column containing URLs (default: 'url').",
    )
    parser.add_argument(
        "--label-col", default="label",
        help="Name of the CSV column containing labels (default: 'label').",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data reserved for testing (default: 0.2).",
    )
    args = parser.parse_args()

    print("="*60)
    print("  Q-Sentinel — Training Pipeline")
    print("="*60)

    # 1. Load data
    print("\n[1/3] Loading dataset...")
    try:
        df = load_dataset(args.data, url_col=args.url_col, label_col=args.label_col)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)
    print(f"  Loaded {len(df)} labelled URLs.")

    # 2. Train & evaluate
    print("\n[2/3] Training model...")
    try:
        metrics = train_and_evaluate(df, test_size=args.test_size)
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    # 3. Report
    print("\n[3/3] Evaluation results")
    print("-"*40)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  FPR       : {metrics['false_positive_rate']:.4f}")
    print(f"  Train size: {metrics['num_train']}")
    print(f"  Test size : {metrics['num_test']}")
    print()
    print(metrics["classification_report"])
    print("="*60)
    print("  Training complete. Model saved to models/model.joblib")
    print("  Run:  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
