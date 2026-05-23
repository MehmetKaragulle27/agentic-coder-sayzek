# SAYZEK Security Architecture

This project extends the SAYZEK agentic coding platform with a local security gateway for LLM-based software generation workflows.

The security layer is designed to inspect:

- user prompts before they reach the LLM,
- LLM outputs before they are returned or executed,
- generated tool/action requests before execution,
- web and SQL-like payloads that may contain attacks.

## Core Components

### InputGuard

Checks user prompts for prompt injection, jailbreak attempts, policy bypass attempts, system prompt extraction, and credential exfiltration requests.

### OutputGuard

Scans model outputs for sensitive data leakage such as API keys, credentials, private keys, database URLs, e-mail addresses, phone numbers, and other risky disclosures.

### SecretScanner

Provides pattern-based detection of secrets and sensitive tokens.

### SecurityPolicy

Controls tool and action execution. It blocks dangerous actions such as reading `.env`, accessing `/etc/passwd`, using unsafe shell pipes, or executing untrusted remote scripts.

### SecurityGateway

Combines input validation, output scanning, and action policy checks into a single gateway interface.

### ModelGuard

Provides model-based security classification support for prompt injection and web attack detection.

## Local Demo

The Streamlit application under `app/` provides a local UI for testing:

- Prompt Injection Guard
- Web Attack Guard
- Batch testing
- Sample payloads

Model artifacts are intentionally excluded from Git because they are large. See `docs/MODEL_ARTIFACTS.md`.
