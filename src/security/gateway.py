from __future__ import annotations

from .input_guard import InputGuard
from .model_guard import LocalModelGuard
from .output_guard import OutputGuard
from .policy import SecurityPolicy
from .schemas import (
    InputSecurityRequest,
    OutputSecurityRequest,
    SecurityDecision,
    SecurityResult,
    ToolActionRequest,
)


class SecurityGateway:
    """
    Central local security gateway for agentic code generation.

    It combines:
    - deterministic input guard
    - deterministic output DLP
    - tool/action policy engine
    - optional H200-hosted local security model
    """

    def __init__(self):
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.policy = SecurityPolicy()
        self.model_guard = LocalModelGuard()

    def check_input(self, user_prompt: str, **metadata) -> SecurityResult:
        request = InputSecurityRequest(
            user_prompt=user_prompt,
            metadata=metadata,
        )

        rule_result = self.input_guard.check(request)
        if rule_result.decision == SecurityDecision.BLOCK:
            return rule_result

        model_result = self.model_guard.classify_text(user_prompt)
        if model_result.decision == SecurityDecision.BLOCK:
            return model_result

        return rule_result

    def check_output(self, llm_output: str, **metadata) -> SecurityResult:
        request = OutputSecurityRequest(
            llm_output=llm_output,
            metadata=metadata,
        )

        dlp_result = self.output_guard.check(request)
        if dlp_result.decision == SecurityDecision.BLOCK:
            return dlp_result

        model_result = self.model_guard.classify_text(llm_output)
        if model_result.decision == SecurityDecision.BLOCK:
            return model_result

        if dlp_result.decision == SecurityDecision.WARN:
            return dlp_result

        return dlp_result

    def check_tool_action(
        self,
        tool_name: str,
        action: str,
        arguments: dict | None = None,
        **metadata,
    ) -> SecurityResult:
        request = ToolActionRequest(
            tool_name=tool_name,
            action=action,
            arguments=arguments or {},
            metadata=metadata,
        )

        return self.policy.check_tool_action(request)
