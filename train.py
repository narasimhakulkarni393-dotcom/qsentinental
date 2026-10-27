"""Q-Sentinel — Model Training and Benchmarking CLI.

Supports:
- PhiUSIIL Phishing URL Websites Dataset (auto-detects and inverts labels: 0=legit -> 0, 1=phish -> 1).
- Generic URL CSV datasets.
- Multiple model architectures: LightGBM, Random Forest, Logistic Regression.
- Domain-Disjoint and Random Stratified evaluation splits.
- Threshold tuning for high phishing recall (reducing False Negatives).
- Optional comparative benchmarking across all models.

Usage:
------
    python train.py
    python train.py --data path/to/dataset.csv --nrows 50000
    python train.py --model-type random_forest --split-mode domain_disjoint
    python train.py --benchmark
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.model import MODEL_DIR, RESULTS_DIR, train_and_evaluate

DEFAULT_PHIUSIIL_PATH = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "kaggleprollc"
    / "phishing-url-websites-dataset-phiusiil"
    / "versions"
    / "1"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def run_benchmark(df: pd.DataFrame) -> dict:
    """Train and compare LightGBM, Random Forest, and Logistic Regression on both splits."""
    print("\n" + "=" * 65)
    print("  RUNNING COMPREHENSIVE BENCHMARK (Classical Models & Split Modes)")
    print("=" * 65)

    comparison = {}
    models = ["lightgbm", "random_forest", "logistic_regression"]
    splits = ["domain_disjoint", "random_stratified"]

    for split in splits:
        comparison[split] = {}
        for m_type in models:
            print(f"\n---> Evaluating [{m_type.upper()}] with [{split.upper()}] split...")
            try:
                metrics = train_and_evaluate(
                    df,
                    model_type=m_type,
                    split_mode=split,
                    target_recall=0.95,
                    max_fpr=0.05,
                    model_path=MODEL_DIR / f"temp_{m_type}_{split}",
                    results_path=RESULTS_DIR / f"temp_{m_type}_{split}",
                )
                comparison[split][m_type] = {
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "f2": metrics["f2"],
                    "fpr": metrics["false_positive_rate"],
                    "false_negatives": metrics["false_negatives"],
                    "false_positives": metrics["false_positives"],
                    "calibrated_threshold": metrics["threshold"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "latency_ms": metrics["inference_latency_ms"],
                }
            except Exception as e:
                print(f"  Error training {m_type} on {split}: {e}")
                comparison[split][m_type] = {"error": str(e)}

    # Save benchmark
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_file = RESULTS_DIR / "benchmark_comparison.json"
    with open(bench_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    print(f"{'Split Mode':<18} | {'Model':<20} | {'Recall':<8} | {'Precision':<10} | {'F1':<6} | {'FPR':<6}")
    print("-" * 75)
    for split, m_dict in comparison.items():
        for m_name, res in m_dict.items():
            if "error" in res:
                print(f"{split:<18} | {m_name:<20} | ERROR: {res['error']}")
            else:
                print(
                    f"{split:<18} | {m_name:<20} | {res['recall']:<8.4f} | {res['precision']:<10.4f} | {res['f1']:<6.4f} | {res['fpr']:<6.4f}"
                )
    print("=" * 65)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Q-Sentinel URL phishing classifier.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_PHIUSIIL_PATH) if DEFAULT_PHIUSIIL_PATH.is_file() else None,
        help="Path to dataset CSV (defaults to PhiUSIIL if found).",
    )
    parser.add_argument(
        "--url-col",
        default="URL",
        help="Name of URL column (default: 'URL').",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of label column (default: 'label').",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "random_forest", "logistic_regression", "hist_gradient_boosting"],
        help="Classifier algorithm (default: 'lightgbm').",
    )
    parser.add_argument(
        "--split-mode",
        default="domain_disjoint",
        choices=["domain_disjoint", "random_stratified"],
        help="Split strategy (default: 'domain_disjoint' to test unseen domains).",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target phishing recall for threshold calibration (default: 0.95).",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Max allowable false positive rate during calibration (default: 0.05).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows to load (e.g. 50000 for faster training).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comparative benchmark of all classical models and splits.",
    )
    args = parser.parse_args()

    if not args.data:
        print("ERROR: No dataset specified and default PhiUSIIL path not found.")
        print("Please provide --data <path-to-csv>")
        sys.exit(1)

    print("=" * 65)
    print("  Q-SENTINEL — PRODUCTION TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Dataset: {args.data}")
    print(f"  Rows limit: {args.nrows if args.nrows else 'Full dataset'}")
    print(f"  Model Type: {args.model_type}")
    print(f"  Split Mode: {args.split_mode}")
    print(f"  Target Recall: {args.target_recall * 100:.1f}%")

    # 1. Load data
    print("\n[1/3] Loading and validating dataset...")
    try:
        df = load_dataset(
            args.data,
            url_col=args.url_col,
            label_col=args.label_col,
            nrows=args.nrows,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)
    print(f"  Loaded {len(df)} validated URLs.")

    # 2. Benchmark option
    if args.benchmark:
        run_benchmark(df)

    # 3. Train primary production model
    print(f"\n[2/3] Training production model [{args.model_type.upper()}] with [{args.split_mode}] split...")
    try:
        metrics = train_and_evaluate(
            df,
            model_type=args.model_type,
            split_mode=args.split_mode,
            target_recall=args.target_recall,
            max_fpr=args.max_fpr,
            model_path=MODEL_DIR,
            results_path=RESULTS_DIR,
        )
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    # 4. Final summary
    print("\n[3/3] Final Test Evaluation Results")
    print("-" * 45)
    print(f"  Model Architecture     : {metrics['model_type']}")
    print(f"  Split Strategy         : {metrics['split_mode']}")
    print(f"  Calibrated Threshold   : {metrics['threshold']:.3f}")
    print(f"  Phishing Recall        : {metrics['recall_phishing'] * 100:.2f}%  (Catches Phishing)")
    print(f"  Precision              : {metrics['precision'] * 100:.2f}%")
    print(f"  F1-Score               : {metrics['f1']:.4f}")
    print(f"  False Positive Rate    : {metrics['false_positive_rate'] * 100:.2f}%")
    print(f"  False Negatives        : {metrics['false_negatives']} (Missed Phishing URLs)")
    print(f"  False Positives        : {metrics['false_positives']} (Legitimate flagged)")
    print(f"  ROC-AUC                : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC                 : {metrics['pr_auc']:.4f}")
    print(f"  Inference Latency      : {metrics['inference_latency_ms']:.2f} ms")
    print()
    print(metrics["classification_report"])
    print("=" * 65)
    print(f"  Model bundle saved to: {MODEL_DIR / 'model.joblib'}")
    print(f"  Metrics saved to:      {RESULTS_DIR / 'metrics.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
