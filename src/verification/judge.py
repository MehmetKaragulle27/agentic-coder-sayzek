"""LLM-as-Judge for filtering false-positive SAST findings."""

import re
import json
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .models import Finding, GateResult, JudgeVerdict, Severity

JUDGE_SYSTEM_PROMPT = """You are a senior security code reviewer. Your job is to triage static analysis findings and classify each one.

For each finding you MUST output a JSON object with exactly these fields:
- "index": the finding number (starting from 0)
- "verdict": one of "true_positive", "false_positive", or "uncertain"
- "reasoning": a brief (1-2 sentence) explanation

Think step by step. Consider:
1. Does the flagged code actually represent a real vulnerability in context?
2. Is the code in a test file where the pattern is intentional (e.g. testing error handling)?
3. Could the finding be triggered by a safe coding pattern that merely resembles a vulnerability?

Output a JSON array of verdict objects. Nothing else."""

JUDGE_PROMPT_TEMPLATE = """Below is code that was flagged by static analysis tools. Review each finding and classify it.

## Code under review
```python
{code}
```

## Findings to triage
{findings_text}

Respond with ONLY a JSON array of verdict objects."""


def _extract_json(text: str):
    """Best-effort JSON extractor that tolerates common model quirks.

    - strips <think>...</think> blocks (Qwen / R1 reasoning models)
    - strips ```json ... ``` fences
    - takes the outermost [...] if present, else the outermost {...}
    - returns None on unrecoverable parse failure
    """
    if not text:
        return None

    # Reasoning models emit their chain-of-thought before the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    return None


class SastJudge:
    """LLM-based judge for SAST false-positive filtering."""

    # Severities the judge triages and which block the gate if confirmed
    # as true positives. Keep in sync with SastAnalyzer.BLOCKING_SEVERITIES.
    TRIAGE_SEVERITIES = (Severity.ERROR, Severity.WARNING)

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _format_findings(self, findings: List[Finding]) -> str:
        """Format findings as numbered text for the prompt."""
        lines = []
        for i, f in enumerate(findings):
            line_info = f" (line {f.line})" if f.line else ""
            code_info = f" [{f.code}]" if f.code else ""
            lines.append(
                f"Finding {i}{code_info}{line_info}: "
                f"[{f.severity.value.upper()}] {f.message}"
            )
        return "\n".join(lines)

    def _parse_verdicts(self, response_text: str, count: int) -> List[JudgeVerdict]:
        """Parse the LLM response into verdict values.

        Handles JSON drift across providers: markdown code fences, extra text
        before/after the JSON, and single-object responses (some models return
        a single {} when there is only one finding).
        Falls back to UNCERTAIN for any entries that fail to parse.
        """
        verdicts = [JudgeVerdict.UNCERTAIN] * count
        items = _extract_json(response_text)
        if items is None:
            return verdicts

        verdict_map = {
            "true_positive": JudgeVerdict.TRUE_POSITIVE,
            "false_positive": JudgeVerdict.FALSE_POSITIVE,
            "uncertain": JudgeVerdict.UNCERTAIN,
        }

        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return verdicts

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            idx = item.get("index", i)
            raw_verdict = str(item.get("verdict", "")).lower().strip()
            if isinstance(idx, int) and 0 <= idx < count:
                verdicts[idx] = verdict_map.get(raw_verdict, JudgeVerdict.UNCERTAIN)

        return verdicts

    def triage(self, code: str, gate_result: GateResult) -> GateResult:
        """Triage SAST findings using the LLM judge.

        Args:
            code: The code that was analyzed
            gate_result: Original SAST gate result

        Returns:
            Updated GateResult with judge verdicts applied
        """
        actionable = [
            f for f in gate_result.findings
            if f.severity in self.TRIAGE_SEVERITIES
            and "not installed" not in f.message
        ]

        if not actionable:
            return gate_result

        findings_text = self._format_findings(actionable)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            code=code,
            findings_text=findings_text,
        )

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        verdicts = self._parse_verdicts(response_text, len(actionable))

        actionable_idx = 0
        updated_findings: List[Finding] = []
        for f in gate_result.findings:
            if f in actionable:
                updated = f.model_copy(update={"judge_verdict": verdicts[actionable_idx]})
                updated_findings.append(updated)
                actionable_idx += 1
            else:
                updated_findings.append(f)

        # Gate fails if any blocking-severity finding survives the
        # false-positive filter (was previously error-only — which let
        # WARNING-severity CWEs like SQL injection pass through).
        has_real_blocking = any(
            f.severity in self.TRIAGE_SEVERITIES
            and f.judge_verdict != JudgeVerdict.FALSE_POSITIVE
            for f in updated_findings
            if "not installed" not in f.message
        )

        return GateResult(
            gate_name=gate_result.gate_name,
            passed=not has_real_blocking,
            findings=updated_findings,
            details=gate_result.details,
        )
