from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-evaluate trained prompt injection classifiers.")

    parser.add_argument("--model-dir", required=True, help="Path to trained model directory.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["neuralchemy", "deepset", "geekyrakshit"],
        help="Evaluation dataset.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--report-path", required=True)

    return parser.parse_args()


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    text = str(value).strip().lower()

    unsafe_values = {
        "1",
        "true",
        "injection",
        "prompt_injection",
        "attack",
        "unsafe",
        "malicious",
    }

    safe_values = {
        "0",
        "false",
        "legit",
        "benign",
        "safe",
        "normal",
    }

    if text in unsafe_values:
        return 1

    if text in safe_values:
        return 0

    raise ValueError(f"Unknown label value: {value!r}")


def load_eval_dataset(dataset_name: str, split: str):
    if dataset_name == "neuralchemy":
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", "full")
        text_col = "text"
        label_col = "label"

    elif dataset_name == "deepset":
        ds = load_dataset("deepset/prompt-injections")
        text_col = "text"
        label_col = "label"

    elif dataset_name == "geekyrakshit":
        ds = load_dataset("geekyrakshit/prompt-injection-dataset")
        text_col = "prompt"
        label_col = "label"

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if split not in ds:
        raise ValueError(f"Split {split!r} not found. Available splits: {list(ds.keys())}")

    return ds[split], text_col, label_col


def main() -> None:
    args = parse_args()

    dataset, text_col, label_col = load_eval_dataset(args.dataset, args.split)

    if args.max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_samples, len(dataset))))

    print("[INFO] Evaluation config:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("[INFO] Dataset rows:", len(dataset))
    print("[INFO] Text column:", text_col)
    print("[INFO] Label column:", label_col)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds: list[int] = []
    labels: list[int] = []

    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        batch = dataset[start:end]

        texts = [str(x) for x in batch[text_col]]
        y = [normalize_label(x) for x in batch[label_col]]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=args.max_length,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()

        preds.extend(batch_preds)
        labels.extend(y)

        if end % (args.batch_size * 100) == 0:
            print(f"[INFO] Evaluated {end}/{len(dataset)}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()

    report = {
        "model_dir": args.model_dir,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(dataset),
        "metrics": {
            "accuracy": float(accuracy),
            "precision_unsafe": float(precision),
            "recall_unsafe": float(recall),
            "f1_unsafe": float(f1),
        },
        "confusion_matrix": cm,
        "label_mapping": {
            "0": "safe",
            "1": "unsafe",
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("[RESULT]")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[DONE] Report written to: {report_path}")


if __name__ == "__main__":
    main()
