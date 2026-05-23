# Model Artifacts

Large model files are excluded from Git.

This project uses locally trained Transformer-based security models.
The actual model weight files are not committed to GitHub because
they are large and can exceed GitHub file size limits.

Ignored artifact examples:

- model.safetensors
- optimizer.pt
- training_args.bin
- Hugging Face cache files
- checkpoint folders

---

## Prompt Guard Models

These models were trained for prompt injection, jailbreak,
system prompt leakage, and unsafe prompt detection.

Local paths used during development:

- Güvenlik/geekyrakshit/models/geekyrakshit-distilbert-prompt-guard
- Güvenlik/neuralchemy/models/neuralchemy-distilbert-prompt-guard

---

## Web Attack Guard Models

These models were trained for web attack detection, including:

- SQL Injection
- XSS
- LFI / path traversal
- encoded payloads
- stored SQL Injection
- obfuscated SQL Injection

Local paths used during development:

- Güvenlik/web_attack/models/web-attack-distilbert-guard
- Güvenlik/web_attack/models/web-attack-distilbert-guard-v2

Current recommended local model:

- Güvenlik/web_attack/models/web-attack-distilbert-guard-v2

---

## SQLShield Experimental Model

This model was trained on the SQLShield dataset as an experimental
SQL risk classifier.

Local path used during development:

- Güvenlik/sql_injection_guard/models/sqlshield-distilbert-sql-guard

This model achieved very high in-dataset performance, but manual tests
showed false positives on simple benign SQL fragments such as WHERE,
SELECT, AND, OR, and simple normal SQL queries.

Therefore, it is kept as an experimental model rather than the final
deployed model.

---

## Why model files are excluded

GitHub has strict file size limits, and trained model artifacts are large.
For example, DistilBERT model files are usually around 250 MB, and
optimizer checkpoints can exceed 500 MB.

For that reason, this repository only tracks:

- source code
- training scripts
- evaluation reports
- documentation
- test files

The actual model weights should be stored locally, on Hugging Face Hub,
or with Git LFS if public model distribution is required.

---

## Expected Local Artifact Structure

A local development machine may contain the following ignored folders:

- Güvenlik/
- Guvenlik/

These folders may include:

- models/
- runs/
- logs/
- data/
- raw/
- backups/
- _hf_cache/

These folders are intentionally ignored by Git.
