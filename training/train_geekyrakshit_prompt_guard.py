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
    parser = argparse.ArgumentParser(description="Train prompt injection classifier on geekyrakshit dataset.")

    parser.add_argument(
        "--dataset-name",
        default="geekyrakshit/prompt-injection-dataset",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Base model for sequence classification.",
    )
    parser.add_argument(
        "--output-dir",
        default="Güvenlik/geekyrakshit/models/geekyrakshit-distilbert-prompt-guard",
        help="Where to save the final model.",
    )
    parser.add_argument(
        "--run-dir",
        default="Güvenlik/geekyrakshit/runs/geekyrakshit-distilbert-run",
        help="Where to save checkpoints/logs.",
    )
    parser.add_argument(
        "--report-path",
        default="Güvenlik/geekyrakshit/reports/geekyrakshit_distilbert_eval_report.json",
        help="Where to save eval metrics.",
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for faster evaluation.",
    )

    return parser.parse_args()


def normalize_label(value: Any) -> int:
    """
    Internal mapping:
    0 = safe / benign
    1 = unsafe / prompt injection
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    text = str(value).strip().lower()

    unsafe_values = {"1", "true", "injection", "prompt_injection", "attack", "unsafe", "malicious"}
    safe_values = {"0", "false", "legit", "benign", "safe", "normal"}

    if text in unsafe_values:
        return 1

    if text in safe_values:
        return 0

    raise ValueError(f"Unknown label value: {value!r}")


def find_text_column(columns: list[str]) -> str:
    for candidate in ["prompt", "text", "input", "content", "instruction"]:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find text column. Available columns: {columns}")


def find_label_column(columns: list[str]) -> str:
    for candidate in ["label", "labels", "target", "is_injection", "is_malicious"]:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find label column. Available columns: {columns}")


def load_geekyrakshit_dataset(dataset_name: str) -> DatasetDict:
    print(f"[INFO] Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print("[INFO] Raw dataset:")
    print(dataset)

    if "train" not in dataset or "test" not in dataset:
        raise ValueError(f"Expected train/test splits. Got: {list(dataset.keys())}")

    # Create validation split from train.
    split = dataset["train"].train_test_split(
        test_size=0.05,
        seed=42,
        shuffle=True,
    )

    prepared = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": dataset["test"],
        }
    )

    print("[INFO] Prepared dataset:")
    print(prepared)

    return prepared


def maybe_limit_dataset(dataset: DatasetDict, max_train_samples: int | None, max_eval_samples: int | None) -> DatasetDict:
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(min(max_train_samples, len(dataset["train"]))))

    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(min(max_eval_samples, len(dataset["validation"]))))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(min(max_eval_samples, len(dataset["test"]))))

    return dataset


def prepare_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    first_split = next(iter(dataset.keys()))
    columns = dataset[first_split].column_names

    text_col = find_text_column(columns)
    label_col = find_label_column(columns)

    print(f"[INFO] Text column: {text_col}")
    print(f"[INFO] Label column: {label_col}")

    def preprocess(batch):
        texts = [str(x) for x in batch[text_col]]
        tokenized = tokenizer(texts, truncation=True, max_length=max_length)
        tokenized["labels"] = [normalize_label(x) for x in batch[label_col]]
        return tokenized

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset[first_split].column_names,
        desc="Tokenizing dataset",
    )

    return tokenized


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
        "precision_unsafe": float(precision),
        "recall_unsafe": float(recall),
        "f1_unsafe": float(f1),
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
        "label_mapping": {"0": "safe", "1": "unsafe"},
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[INFO] Report written to: {report_path}")


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Training configuration:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    dataset = load_geekyrakshit_dataset(args.dataset_name)
    dataset = maybe_limit_dataset(dataset, args.max_train_samples, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = prepare_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1},
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_unsafe",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=False,
        bf16=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Evaluating best model...")
    metrics = trainer.evaluate(tokenized["test"])
    print("[RESULT] Test metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"[INFO] Saving model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    save_report(args.report_path, metrics, trainer, tokenized["test"])

    print("[DONE] Geekyrakshit prompt guard training completed.")


if __name__ == "__main__":
    main()
