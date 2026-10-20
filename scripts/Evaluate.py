python
"""
Aggregate baseline and transformer metrics into a single summary table.

Reads:
    results/baseline_metrics.json
    results/transformer_metrics.json

Writes:
    results/summary_results.csv

Usage:
    python scripts/evaluate.py
"""

import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path("results")


def load_json(path):
    with path.open() as f:
        return json.load(f)


def main():
    baseline_path = OUT_DIR / "baseline_metrics.json"
    transformer_path = OUT_DIR / "transformer_metrics.json"

    rows = []

    if baseline_path.exists():
        baseline = load_json(baseline_path)
        for key, res in baseline.items():
            task = key.split("__")[0]
            rows.append({
                "Task": task,
                "Model": res["model"],
                "Precision": res["macro_precision"],
                "Recall": res["macro_recall"],
                "Macro-F1": res["macro_f1"],
                "Accuracy": res["accuracy"],
            })
    else:
        print(f"[warn] {baseline_path} not found — skipping baselines.")

    if transformer_path.exists():
        transformer = load_json(transformer_path)
        for task, res in transformer.items():
            rows.append({
                "Task": task,
                "Model": "BERT (bert-base-uncased)",
                "Precision": res.get("eval_macro_precision"),
                "Recall": res.get("eval_macro_recall"),
                "Macro-F1": res.get("eval_macro_f1"),
                "Accuracy": res.get("eval_accuracy"),
            })
    else:
        print(f"[warn] {transformer_path} not found — skipping transformer.")

    if not rows:
        print("No results found. Run train_baselines.py and/or "
              "train_transformer.py first.")
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "summary_results.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Summary Table ===")
    print(df.to_string(index=False))
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
