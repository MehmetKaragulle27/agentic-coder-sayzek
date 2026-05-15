from __future__ import annotations

import re
from typing import List

from .schemas import SecurityCategory, SecurityFinding, SecuritySeverity


class SecretScanner:
    """Fast deterministic scanner for secrets and PII-like values."""

    SECRET_PATTERNS = [
        ("openai_api_key", r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
        ("anthropic_api_key", r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
        ("github_token", r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
        ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
        ("jwt", r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
        ("private_key", r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        ("password", r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
        ("secret", r"(?i)\b(secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|bearer_token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        ("database_url_with_creds", r"(?i)\b(postgres|postgresql|mysql|mongodb|redis)://[^:\s]+:[^@\s]+@[^ \n\r\t]+"),
    ]

    PII_PATTERNS = [
        ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        ("phone", r"(?<!\d)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}(?!\d)"),
    ]

    def scan(self, text: str) -> List[SecurityFinding]:
        text = text or ""
        findings: List[SecurityFinding] = []

        for name, pattern in self.SECRET_PATTERNS:
            for match in re.finditer(pattern, text):
                findings.append(
                    SecurityFinding(
                        category=SecurityCategory.SECRET_LEAKAGE,
                        severity=SecuritySeverity.CRITICAL,
                        message=f"Secret detected: {name}",
                        rule_id=f"secret_{name}",
                        metadata={
                            "start": match.start(),
                            "end": match.end(),
                            "preview": self._preview(match.group(0)),
                        },
                    )
                )

        for name, pattern in self.PII_PATTERNS:
            for match in re.finditer(pattern, text):
                findings.append(
                    SecurityFinding(
                        category=SecurityCategory.PII_LEAKAGE,
                        severity=SecuritySeverity.MEDIUM,
                        message=f"Potential PII detected: {name}",
                        rule_id=f"pii_{name}",
                        metadata={
                            "start": match.start(),
                            "end": match.end(),
                            "preview": self._preview(match.group(0)),
                        },
                    )
                )

        return findings

    def _preview(self, value: str) -> str:
        value = str(value).replace("\n", "\\n")
        if len(value) <= 8:
            return "[REDACTED]"
        return value[:3] + "..." + value[-3:]
