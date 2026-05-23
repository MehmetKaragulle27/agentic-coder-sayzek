from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_WEB_ATTACK_MODEL_PATH = "Guvenlik/web_attack/models/web-attack-distilbert-guard"


@dataclass
class WebAttackResult:
    text: str
    predicted_label: int
    decision: str
    action: str
    benign_score: float
    attack_score: float
    confidence: float
    model_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "decision": self.decision,
            "action": self.action,
            "benign_score": self.benign_score,
            "attack_score": self.attack_score,
            "confidence": self.confidence,
            "model_path": self.model_path,
        }


class WebAttackService:
    """
    Local web attack classifier.

    Internal label mapping:
    0 = benign
    1 = web_attack
    """

    def __init__(
        self,
        model_path: str = DEFAULT_WEB_ATTACK_MODEL_PATH,
        max_length: int = 256,
        attack_threshold: float = 0.50,
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.attack_threshold = attack_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, text: str) -> WebAttackResult:
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

        benign_score = float(probs[0].item())
        attack_score = float(probs[1].item())

        predicted_label = 1 if attack_score >= self.attack_threshold else 0

        if predicted_label == 1:
            decision = "WEB_ATTACK"
            action = "BLOCK"
            confidence = attack_score
        else:
            decision = "BENIGN"
            action = "ALLOW"
            confidence = benign_score

        return WebAttackResult(
            text=text,
            predicted_label=predicted_label,
            decision=decision,
            action=action,
            benign_score=benign_score,
            attack_score=attack_score,
            confidence=confidence,
            model_path=self.model_path,
        )