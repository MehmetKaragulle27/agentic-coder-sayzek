from __future__ import annotations

import re

from .schemas import (
    InputSecurityRequest,
    SecurityCategory,
    SecurityDecision,
    SecurityFinding,
    SecurityResult,
    SecuritySeverity,
)


class InputGuard:
    """Input guard for prompt injection, jailbreak and credential exfiltration."""

    PATTERNS = [
        (
            "prompt_injection",
            SecurityCategory.PROMPT_INJECTION,
            r"(?i)\b(ignore|forget|disregard|override)\b.{0,120}\b(previous|above|system|developer)\b.{0,120}\b(instruction|prompt|message|rules?)\b",
        ),
        (
            "system_prompt_leakage",
            SecurityCategory.SYSTEM_PROMPT_LEAKAGE,
            r"(?i)\b(reveal|show|print|dump|expose|leak|display)\b.{0,120}\b(system prompt|developer message|hidden prompt|internal instructions|policy)\b",
        ),
        (
            "jailbreak",
            SecurityCategory.JAILBREAK,
            r"(?i)\b(jailbreak|DAN|do anything now|developer mode|evil mode|god mode|unrestricted mode)\b",
        ),
        (
            "policy_bypass",
            SecurityCategory.POLICY_BYPASS,
            r"(?i)\b(disable|turn off|remove|bypass|ignore)\b.{0,120}\b(safety|security|guardrail|filter|validation|sandbox|policy)\b",
        ),
        (
            "credential_exfiltration",
            SecurityCategory.CREDENTIAL_EXFILTRATION,
            r"(?i)(\b(api[_ -]?keys?|tokens?|passwords?|secrets?|private keys?|ssh keys?|credentials?)\b.{0,120}\b(show|print|reveal|dump|exfiltrate|steal|leak)\b|\b(show|print|reveal|dump|exfiltrate|steal|leak)\b.{0,120}\b(api[_ -]?keys?|tokens?|passwords?|secrets?|private keys?|ssh keys?|credentials?)\b)",
        ),
        (
            "explicit_prompt_injection",
            SecurityCategory.PROMPT_INJECTION,
            r"(?i)\b(prompt injection|indirect prompt injection|instruction override|system override)\b",
        ),
    ]

    def check(self, request: InputSecurityRequest) -> SecurityResult:
        prompt = request.user_prompt or ""
        findings = []

        for rule_id, category, pattern in self.PATTERNS:
            if re.search(pattern, prompt):
                findings.append(
                    SecurityFinding(
                        category=category,
                        severity=SecuritySeverity.HIGH,
                        message=f"Blocked unsafe input by rule: {rule_id}",
                        rule_id=rule_id,
                    )
                )

        if findings:
            return SecurityResult(
                decision=SecurityDecision.BLOCK,
                safe=False,
                findings=findings,
                source="input_guard_rules",
            )

        return SecurityResult(
            decision=SecurityDecision.ALLOW,
            safe=True,
            findings=[],
            source="input_guard_rules",
        )
