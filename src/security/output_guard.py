from __future__ import annotations

from .schemas import (
    OutputSecurityRequest,
    SecurityDecision,
    SecurityResult,
    SecuritySeverity,
)
from .secret_scanner import SecretScanner


class OutputGuard:
    """Output guard for DLP and unsafe generated content."""

    def __init__(self):
        self.secret_scanner = SecretScanner()

    def check(self, request: OutputSecurityRequest) -> SecurityResult:
        findings = self.secret_scanner.scan(request.llm_output)

        has_critical = any(
            finding.severity == SecuritySeverity.CRITICAL
            for finding in findings
        )

        if has_critical:
            return SecurityResult(
                decision=SecurityDecision.BLOCK,
                safe=False,
                findings=findings,
                source="output_guard",
            )

        if findings:
            return SecurityResult(
                decision=SecurityDecision.WARN,
                safe=True,
                findings=findings,
                source="output_guard",
            )

        return SecurityResult(
            decision=SecurityDecision.ALLOW,
            safe=True,
            findings=[],
            source="output_guard",
        )
