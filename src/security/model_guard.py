from __future__ import annotations

import os
from typing import Optional

from .schemas import (
    SecurityCategory,
    SecurityDecision,
    SecurityFinding,
    SecurityResult,
    SecuritySeverity,
)


class LocalModelGuard:
    """
    Local model-based security guard.

    This class is the integration point for H200-hosted local models.

    First target models:
    - meta-llama/Llama-Prompt-Guard-2-86M
    - meta-llama/Llama-Guard-4-12B

    The model is disabled by default. Enable it with:
    LOCAL_SECURITY_MODEL_ENABLED=true
    """

    def __init__(self):
        self.enabled = os.getenv("LOCAL_SECURITY_MODEL_ENABLED", "false").lower() == "true"
        self.model_name = os.getenv(
            "LOCAL_SECURITY_MODEL_NAME",
            "meta-llama/Llama-Prompt-Guard-2-86M",
        )
        self.model = None
        self.tokenizer = None
        self.load_error: Optional[str] = None

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            if not torch.cuda.is_available():
                self.model.to("cpu")

        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.load_error = repr(exc)

    def classify_text(self, text: str) -> SecurityResult:
        """
        Classify text with a local security model.

        If model usage is disabled, this returns ALLOW.
        Deterministic guards still run before/after this model in SecurityGateway.
        """
        if not self.enabled:
            return SecurityResult(
                decision=SecurityDecision.ALLOW,
                safe=True,
                findings=[],
                source="local_model_disabled",
            )

        if self.model is None or self.tokenizer is None:
            return SecurityResult(
                decision=SecurityDecision.WARN,
                safe=True,
                findings=[
                    SecurityFinding(
                        category=SecurityCategory.UNKNOWN_RISK,
                        severity=SecuritySeverity.MEDIUM,
                        message=f"Local security model could not be loaded: {self.load_error}",
                        rule_id="local_model_load_error",
                    )
                ],
                source="local_model_error",
            )

        try:
            import torch

            inputs = self.tokenizer(
                text or "",
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            inputs = {
                key: value.to(self.model.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            score = float(probs.max().item())
            label_id = int(probs.argmax().item())

            # Conservative first implementation:
            # label 0 is treated as safe; non-zero labels are treated as unsafe.
            # We will refine this mapping after validating the selected model's label schema.
            if label_id != 0:
                return SecurityResult(
                    decision=SecurityDecision.BLOCK,
                    safe=False,
                    findings=[
                        SecurityFinding(
                            category=SecurityCategory.PROMPT_INJECTION,
                            severity=SecuritySeverity.HIGH,
                            message="Local security model classified text as unsafe.",
                            rule_id="local_model_unsafe",
                            score=score,
                            metadata={"label_id": label_id},
                        )
                    ],
                    source="local_model",
                    metadata={"label_id": label_id, "score": score},
                )

            return SecurityResult(
                decision=SecurityDecision.ALLOW,
                safe=True,
                findings=[],
                source="local_model",
                metadata={"label_id": label_id, "score": score},
            )

        except Exception as exc:
            return SecurityResult(
                decision=SecurityDecision.WARN,
                safe=True,
                findings=[
                    SecurityFinding(
                        category=SecurityCategory.UNKNOWN_RISK,
                        severity=SecuritySeverity.MEDIUM,
                        message=f"Local security model inference failed: {exc!r}",
                        rule_id="local_model_inference_error",
                    )
                ],
                source="local_model_error",
            )
