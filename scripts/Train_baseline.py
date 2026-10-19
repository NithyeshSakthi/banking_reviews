python
"""
Train classical baselines for ABSA tasks:
    - TF-IDF + Logistic Regression
    - TF-IDF + Linear SVM

Tasks:
    - Aspect sentiment classification (3-class)
    - Fine-grained issue-type classification (multi-class)
    - Reactive/proactive positioning (binary/multi-class)

Outputs:
    results/baseline_metrics.json

Usage:
    python scripts/train_baselines.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

# Make set_seed importable when running from the scripts/ folder
sys.path.append(str(Path(__file__).resolve().parent.parent))
from set_seed import set_seed  # noqa: E402

SEED = 42
set_seed(SEED)

DATA_DIR = Path("data/annotated")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

LR_C = 1.0
LR_MAX_ITER = 1000

SVM_C = 1.0

TASKS = {
    "sentiment": "sentiment_label",
    "issue_type": "issue_type_label",
    "positioning": "positioning_label",
}

TEXT_COL = "text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected split file not found: {path}. "
            f"Ensure data/annotated/{{train,val,test}}.csv exist."
        )
    return pd.read_csv(path)


def evaluate(model_name, model, X_train, y_train, X_test, y_test):
    """Train a model and return a metrics dictionary."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df = load_split("train")
    test_df = load_split("test")

    all_results = {}

    for task_name, label_col in TASKS.items():
        if label_col not in train_df.columns:
            print(f"[skip] Task '{task_name}' — column '{label_col}' not found.")
            continue

        print(f"\n=== Task: {task_name} (label column: {label_col}) ===")

        X_train_text = train_df[TEXT_COL].astype(str).tolist()
        y_train = train_df[label_col].astype(str).tolist()
        X_test_text = test_df[TEXT_COL].astype(str).tolist()
        y_test = test_df[label_col].astype(str).tolist()

        # Fit vectorizer on training data only (no leakage)
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        # Logistic Regression
        lr = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            random_state=SEED,
        )
        all_results[f"{task_name}__logreg"] = evaluate(
            "TF-IDF + LogisticRegression",
            lr, X_train, y_train, X_test, y_test,
        )

        # Linear SVM
        svm = LinearSVC(
            C=SVM_C,
            class_weight="balanced",
            random_state=SEED,
        )
        all_results[f"{task_name}__svm"] = evaluate(
            "TF-IDF + LinearSVC",
            svm, X_train, y_train, X_test, y_test,
        )

    out_path = OUT_DIR / "baseline_metrics.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Task':<14} {'Model':<32} {'Macro-F1':<10} {'Accuracy':<10}")
    print("-" * 70)
    for key, res in all_results.items():
        print(f"{key:<14} {res['model']:<32} {res['macro_f1']:<10} {res['accuracy']:<10}")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()