"""Q-Sentinel — Hybrid quantum-classical training CLI.

Usage
-----
    python train_quantum.py --data data/phishing_site_urls.csv --url-col URL --label-col Label
    python train_quantum.py --data data/dataset.csv --qubits 6 --layers 3 --steps 200
"""

import argparse
import json
import sys
from pathlib import Path

from src.data_loader import load_dataset
from src.quantum_model import train_quantum_and_evaluate


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Q-Sentinel variational quantum classifier.")
    p.add_argument("--data", required=True, help="Labelled CSV file.")
    p.add_argument("--url-col", default="url")
    p.add_argument("--label-col", default="label")
    p.add_argument("--qubits", type=int, default=4, help="Qubits = features used (default 4).")
    p.add_argument("--layers", type=int, default=3, help="Variational layers (default 3).")
    p.add_argument("--steps", type=int, default=120, help="Optimiser steps (default 120).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=800,
                   help="Max samples the quantum circuit trains on (default 800).")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()

    print("=" * 60)
    print("  Q-Sentinel — Hybrid Quantum Training")
    print("=" * 60)

    print("\n[1/3] Loading dataset...")
    try:
        df = load_dataset(args.data, url_col=args.url_col, label_col=args.label_col)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)
    print(f"  Loaded {len(df)} labelled URLs.")

    print("\n[2/3] Training quantum classifier...")
    try:
        m = train_quantum_and_evaluate(
            df, n_qubits=args.qubits, n_layers=args.layers, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, max_train=args.max_train,
            test_size=args.test_size,
        )
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print("\n[3/3] Quantum vs classical (same test split)")
    print("-" * 40)
    classical_file = Path("results") / "metrics.json"
    classical = json.loads(classical_file.read_text()) if classical_file.is_file() else None
    print(f"  {'Metric':<10}{'Quantum':>10}{'Classical':>12}")
    for key, label in [("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("false_positive_rate", "FPR")]:
        c = f"{classical[key]:.4f}" if classical else "n/a"
        print(f"  {label:<10}{m[key]:>10.4f}{c:>12}")
    if classical is None:
        print("  (run train.py first to see the classical column)")
    print()
    print(m["classification_report"])
    print("=" * 60)
    print("  Done. Run:  streamlit run app.py   ->  Quantum Lab")
    print("=" * 60)


if __name__ == "__main__":
    main()
