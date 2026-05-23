from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Web Attack Guard v2 with normal SQL hard negatives.")

    parser.add_argument("--data-path", default="Guvenlik/web_attack/data/web_attack_v2_sql_hardneg.jsonl")
    parser.add_argument("--base-model", default="Guvenlik/web_attack/backups/web-attack-distilbert-guard-v1")
    parser.add_argument("--output-dir", default="Guvenlik/web_attack/models/web-attack-distilbert-guard-v2")
    parser.add_argument("--run-dir", default="Guvenlik/web_attack/runs/web-attack-distilbert-v2-run")
    parser.add_argument("--report-path", default="Guvenlik/web_attack/reports/web_attack_eval_report_v2.json")

    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)

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
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "f1_attack": float(f1),
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
            "0": "benign",
            "1": "web_attack",
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

    print("[INFO] Fine-tuning configuration:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    raw = load_dataset("json", data_files=args.data_path)["train"]

    split1 = raw.train_test_split(test_size=0.20, seed=args.seed, shuffle=True)
    split2 = split1["test"].train_test_split(test_size=0.50, seed=args.seed, shuffle=True)

    ds = DatasetDict(
        {
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"],
        }
    )

    print("[INFO] Dataset:")
    print(ds)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess(batch):
        tokenized = tokenizer(
            [str(x) for x in batch["text"]],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = [int(x) for x in batch["label"]]
        return tokenized

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing v2 dataset",
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)

    training_args = TrainingArguments(
        output_dir=args.run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_attack",
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

    print("[INFO] Starting Web Attack Guard v2 fine-tuning...")
    trainer.train()

    print("[INFO] Evaluating v2 model...")
    metrics = trainer.evaluate(tokenized["test"])

    print("[RESULT]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"[INFO] Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_report(args.report_path, metrics, trainer, tokenized["test"])

    print("[DONE] Web Attack Guard v2 fine-tuning completed.")


if __name__ == "__main__":
    main()
