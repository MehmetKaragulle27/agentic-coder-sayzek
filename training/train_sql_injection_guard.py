from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SQL Injection / SQL Risk Guard with SQLShield.")

    parser.add_argument("--dataset-name", default="salmane11/SQLShield")
    parser.add_argument("--model-name", default="distilbert-base-uncased")

    parser.add_argument("--output-dir", default="Guvenlik/sql_injection_guard/models/sqlshield-distilbert-sql-guard")
    parser.add_argument("--run-dir", default="Guvenlik/sql_injection_guard/runs/sqlshield-distilbert-run")
    parser.add_argument("--report-path", default="Guvenlik/sql_injection_guard/reports/sqlshield_eval_report.json")

    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "precision_sql_risk": float(precision),
        "recall_sql_risk": float(recall),
        "f1_sql_risk": float(f1),
    }


def save_report(path: str, metrics: Dict[str, Any], trainer: Trainer, eval_dataset) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    cm = confusion_matrix(labels, preds).tolist()

    report = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "label_mapping": {
            "0": "benign_sql",
            "1": "sql_risk",
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[INFO] Report written to: {report_path}")


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Training configuration:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    ds = load_dataset(args.dataset_name)

    print("[INFO] Dataset:")
    print(ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(batch):
        tokenized = tokenizer(
            [str(x) for x in batch["query"]],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = [int(x) for x in batch["malicious"]]
        return tokenized

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing SQLShield",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "benign_sql", 1: "sql_risk"},
        label2id={"benign_sql": 0, "sql_risk": 1},
    )

    training_args = TrainingArguments(
        output_dir=args.run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=0.10,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_sql_risk",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Evaluating best model on test set...")
    metrics = trainer.evaluate(tokenized["test"])

    print("[RESULT] Test metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"[INFO] Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_report(args.report_path, metrics, trainer, tokenized["test"])

    print("[DONE] SQL Injection Guard training completed.")


if __name__ == "__main__":
    main()
