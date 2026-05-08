"""Connectivity probe for every endpoint configured in .env.

Reads CODING_PROVIDER/MODEL/FALLBACKS and JUDGE_PROVIDER/MODEL/FALLBACKS
directly, then hits each endpoint with a trivial prompt so you can see which
ones actually work with your current keys.
"""

import os
import sys
from typing import List, Tuple
from dotenv import load_dotenv
import httpx

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROFILES = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",
               "GOOGLE_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "github": ("https://models.inference.ai.azure.com", "GITHUB_TOKEN"),
    "ollama": ("https://ollama.com/v1", "OLLAMA_API_KEY"),
    "opencode": (
        os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"),
        "OPENCODE_API_KEY",
    ),
}


def parse_chain(primary_prov: str, primary_model: str,
                fallback_env: str) -> List[Tuple[str, str]]:
    chain: List[Tuple[str, str]] = []
    if primary_prov and primary_model:
        chain.append((primary_prov, primary_model))
    raw = os.getenv(fallback_env, "")
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        prov, _, model = part.partition(":")
        chain.append((prov.strip(), model.strip()))
    return chain


def probe(role: str, prov: str, model: str) -> str:
    if prov not in PROFILES:
        return f"  [unknown provider: {prov}]"
    base_url, env_key = PROFILES[prov]
    key = os.getenv(env_key)
    tag = f"[{role}] {prov}:{model}"
    if not key:
        return f"  SKIP {tag} ({env_key} not set)"
    try:
        r = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "reply with one word: ok"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=30.0,
        )
        if r.status_code == 200:
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()
            return f"  OK   {tag}  -> {reply!r}"
        body = r.text[:200].replace("\n", " ")
        return f"  FAIL {tag}  HTTP {r.status_code}: {body}"
    except Exception as e:  # noqa: BLE001
        return f"  ERR  {tag}  {type(e).__name__}: {str(e)[:200]}"


if __name__ == "__main__":
    coding = parse_chain(
        os.getenv("CODING_PROVIDER", ""),
        os.getenv("CODING_MODEL", ""),
        "CODING_FALLBACKS",
    )
    judge = parse_chain(
        os.getenv("JUDGE_PROVIDER", ""),
        os.getenv("JUDGE_MODEL", ""),
        "JUDGE_FALLBACKS",
    )

    print("\n=== Probing CODING chain ===")
    for prov, model in coding:
        print(probe("coding", prov, model))

    print("\n=== Probing JUDGE chain ===")
    for prov, model in judge:
        print(probe("judge", prov, model))

    print("\n(OK = reachable, FAIL/ERR = will be skipped at runtime)")
