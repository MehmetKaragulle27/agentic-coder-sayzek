from __future__ import annotations

import re

from .schemas import (
    SecurityCategory,
    SecurityDecision,
    SecurityFinding,
    SecurityResult,
    SecuritySeverity,
    ToolActionRequest,
)


class SecurityPolicy:
    """Policy engine for blocking dangerous agent actions."""

    BLOCKED_PATH_PATTERNS = [
        r"(?i)(^|['\"\s:/])\.env(['\"\s}]|$)",
        r"(?i)(^|['\"\s])[^'\"\s]*\.ssh(/|['\"\s}]|$)",
        r"(?i)(^|/)(id_rsa|id_ed25519)(['\"\s}]|$)",
        r"(?i)(^|/)(secret|secrets|credential|credentials)(\.|/|['\"\s}]|$)",
        r"(?i)/etc/",
        r"(?i)/etc/passwd",
        r"(?i)/root/",
        r"(?i)/home/[^/]+/\.ssh/",
    ]

    BLOCKED_COMMAND_PATTERNS = [
        r"(?i)\bcat\s+\.env\b",
        r"(?i)\bcat\s+/etc/passwd\b",
        r"(?i)\brm\s+-rf\s+/",
        r"(?i)\bcurl\b.*\|\s*(bash|sh)",
        r"(?i)\bwget\b.*\|\s*(bash|sh)",
        r"(?i)\bbase64\b.*\|\s*(bash|sh)",
        r"(?i)\b(nc|netcat)\b",
        r"(?i)\bchmod\s+777\b",
        r"(?i)\bsudo\b",
        r"(?i)\bdocker\b.*--privileged",
    ]

    def check_tool_action(self, request: ToolActionRequest) -> SecurityResult:
        findings = []

        serialized_args = str(request.arguments)
        action_text = f"{request.tool_name} {request.action} {serialized_args}"

        for pattern in self.BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, action_text):
                findings.append(
                    SecurityFinding(
                        category=SecurityCategory.UNSAFE_COMMAND,
                        severity=SecuritySeverity.CRITICAL,
                        message=f"Blocked dangerous command/action pattern: {pattern}",
                        rule_id="blocked_command_pattern",
                    )
                )

        for pattern in self.BLOCKED_PATH_PATTERNS:
            if re.search(pattern, action_text) or re.search(pattern, serialized_args):
                findings.append(
                    SecurityFinding(
                        category=SecurityCategory.DATA_EXFILTRATION,
                        severity=SecuritySeverity.CRITICAL,
                        message=f"Blocked sensitive path access pattern: {pattern}",
                        rule_id="blocked_path_pattern",
                    )
                )

        if findings:
            return SecurityResult(
                decision=SecurityDecision.BLOCK,
                safe=False,
                findings=findings,
                source="policy",
            )

        return SecurityResult(
            decision=SecurityDecision.ALLOW,
            safe=True,
            findings=[],
            source="policy",
        )
