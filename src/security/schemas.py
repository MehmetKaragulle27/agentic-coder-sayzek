from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SecurityDecision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REQUIRE_REVIEW = "require_review"


class SecurityCategory(str, Enum):
    SAFE = "safe"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    CREDENTIAL_EXFILTRATION = "credential_exfiltration"
    TOOL_ABUSE = "tool_abuse"
    SECRET_LEAKAGE = "secret_leakage"
    PII_LEAKAGE = "pii_leakage"
    UNSAFE_COMMAND = "unsafe_command"
    MALICIOUS_CODE = "malicious_code"
    DATA_EXFILTRATION = "data_exfiltration"
    POLICY_BYPASS = "policy_bypass"
    UNKNOWN_RISK = "unknown_risk"


class SecuritySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityFinding(BaseModel):
    category: SecurityCategory
    severity: SecuritySeverity
    message: str
    rule_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecurityResult(BaseModel):
    decision: SecurityDecision
    safe: bool
    findings: List[SecurityFinding] = Field(default_factory=list)
    source: str = "unknown"
    raw_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InputSecurityRequest(BaseModel):
    user_prompt: str
    system_context: Optional[str] = None
    available_tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OutputSecurityRequest(BaseModel):
    llm_output: str
    task_type: Optional[str] = None
    target_file: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolActionRequest(BaseModel):
    tool_name: str
    action: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
