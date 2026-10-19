python
"""
Fine-tune bert-base-uncased for ABSA tasks:
    - Aspect sentiment classification (3-class)
    - Fine-grained issue-type classification (multi-class)
    - Reactive/proactive positioning (binary/multi-class)
    - Aspect-term extraction (token classification, optional)

Outputs:
    results/transformer_metrics.json
    models/{task}/    (checkpoints)

Usage:
    python scripts/train_transformer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from set_seed import set_seed  # noqa: E402

SEED = 42
set_seed(SEED)

DATA_DIR = Path("data/annotated")
OUT_DIR = Path("results")
MODEL_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters (match Section 6 of the dissertation)
# ---------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 4
EARLY_STOPPING_PATIENCE = 2
WEIGHT_DECAY = 0.01

TASKS = {
    "sentiment": "sentiment_label",
    "issue_type": "issue_type_label",
    "positioning": "positioning_label",
}

TEXT_COL = "text"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = [label2id[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def train_task(task_name, label_col, train_df, val_df, test_df):
    labels = sorted(train_df[label_col].astype(str).unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"\n=== Training task: {task_name} ({len(labels)} classes) ===")
    print(f"Classes: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = TextClassificationDataset(
        train_df[TEXT_COL].astype(str).tolist(),
        train_df[label_col].astype(str).tolist(),
        tokenizer, label2id,
    )
    val_ds = TextClassificationDataset(
        val_df[TEXT_COL].astype(str).tolist(),
        val_df[label_col].astype(str).tolist(),
        tokenizer, label2id,
    )
    test_ds = TextClassificationDataset(
        test_df[TEXT_COL].astype(str).tolist(),
        test_df[label_col].astype(str).tolist(),
        tokenizer, label2id,
    )

    # Early stopping via transformers' callback (patience = 2)
    from transformers import EarlyStoppingCallback

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / task_name),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )],
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    rounded = {
        k: round(float(v), 4) if isinstance(v, (int, float)) else v
        for k, v in test_metrics.items()
    }
    rounded["labels"] = labels
    rounded["label2id"] = label2id
    return rounded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    results = {}
    for task_name, label_col in TASKS.items():
        if label_col not in train_df.columns:
            print(f"[skip] Task '{task_name}' — column '{label_col}' not found.")
            continue
        results[task_name] = train_task(
            task_name, label_col, train_df, val_df, test_df
        )

    out_path = OUT_DIR / "transformer_metrics.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'Task':<14} {'Macro-F1':<10} {'Accuracy':<10}")
    print("-" * 40)
    for task, res in results.items():
        print(f"{task:<14} {res.get('eval_macro_f1', 'n/a'):<10} "
              f"{res.get('eval_accuracy', 'n/a'):<10}")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()