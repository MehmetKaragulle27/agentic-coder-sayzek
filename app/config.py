from __future__ import annotations

import os


DEFAULT_MODEL_PATH = "Güvenlik/geekyrakshit/models/geekyrakshit-distilbert-prompt-guard"


class AppConfig:
    model_path: str = os.getenv("PROMPT_GUARD_MODEL_PATH", DEFAULT_MODEL_PATH)
    max_length: int = int(os.getenv("PROMPT_GUARD_MAX_LENGTH", "256"))
    unsafe_threshold: float = float(os.getenv("PROMPT_GUARD_THRESHOLD", "0.50"))
