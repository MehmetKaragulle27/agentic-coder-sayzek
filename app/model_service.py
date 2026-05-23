from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import AppConfig


@dataclass
class PromptGuardResult:
    text: str
    predicted_label: int
    decision: str
    action: str
    safe_score: float
    unsafe_score: float
    confidence: float
    model_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "decision": self.decision,
            "action": self.action,
            "safe_score": self.safe_score,
            "unsafe_score": self.unsafe_score,
            "confidence": self.confidence,
            "model_path": self.model_path,
        }


class PromptGuardService:
    """
    Local prompt security classifier.

    Internal label mapping:
    0 = safe
    1 = unsafe
    """

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.model_path = self.config.model_path
        self.max_length = self.config.max_length
        self.unsafe_threshold = self.config.unsafe_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text: str) -> PromptGuardResult:
        text = text or ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        safe_score = float(probs[0].item())
        unsafe_score = float(probs[1].item())

        predicted_label = 1 if unsafe_score >= self.unsafe_threshold else 0

        if predicted_label == 1:
            decision = "UNSAFE"
            action = "BLOCK"
            confidence = unsafe_score
        else:
            decision = "SAFE"
            action = "ALLOW"
            confidence = safe_score

        return PromptGuardResult(
            text=text,
            predicted_label=predicted_label,
            decision=decision,
            action=action,
            safe_score=safe_score,
            unsafe_score=unsafe_score,
            confidence=confidence,
            model_path=self.model_path,
        )
