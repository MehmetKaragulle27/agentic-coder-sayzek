"""Benchmark latency for Ollama Cloud models available to your account.

This script:
1) Calls GET /models on Ollama's OpenAI-compatible endpoint.
2) Filters to cloud models by default (model id contains "cloud").
3) Sends a tiny chat-completions request N times per model.
4) Measures:
   - time-to-first-byte (ttfb_ms): request send -> first response byte
   - total latency (total_ms): request send -> full body received
5) Writes both CSV and Markdown reports under reports/.

Usage examples:
  python scripts/benchmark_ollama_cloud_latency.py
  python scripts/benchmark_ollama_cloud_latency.py --runs 3 --timeout 180
  python scripts/benchmark_ollama_cloud_latency.py --model "glm-5.1:cloud"
  python scripts/benchmark_ollama_cloud_latency.py --include-non-cloud
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_PROMPT = "Reply with exactly: ok"

PROMPT_PRESETS = {
    # Tiny ping-style latency probe.
    "tiny": DEFAULT_PROMPT,
    # Benchmark-like unit-test generation workload (closest to your pipeline).
    "ult_like": (
        "Generate comprehensive unit tests for the function below.\n\n"
        "Problem description:\n"
        "Given a list of integers, return True if there exist two distinct "
        "indices i, j such that nums[i] + nums[j] == target.\n"
        "Handle duplicates, negative values, and empty input.\n\n"
        "Code under test:\n"
        "def has_two_sum(nums, target):\n"
        "    seen = set()\n"
        "    for x in nums:\n"
        "        if target - x in seen:\n"
        "            return True\n"
        "        seen.add(x)\n"
        "    return False\n\n"
        "Write high-quality pytest tests only."
    ),
    # Security/CWE-ish benchmark workload.
    "cweval_like": (
        "Generate comprehensive unit tests for this function and include "
        "malicious/boundary inputs.\n\n"
        "Code under test:\n"
        "def unsafe_login(user, password):\n"
        "    q = \"SELECT * FROM users WHERE user='\" + user + \"' AND pass='\" + password + \"'\"\n"
        "    return q\n\n"
        "Focus on SQL injection-like payloads, escaping edge cases, empty/null "
        "inputs, unicode, and long strings. Output pytest test code only."
    ),
}


@dataclass
class RunSample:
    model: str
    run_idx: int
    ok: bool
    status_code: int | None
    ttfb_ms: float | None
    total_ms: float | None
    response_chars: int
    error: str


def _ensure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _short_err(err: str, limit: int = 220) -> str:
    text = (err or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_models_from_openai_payload(data: dict) -> list[str]:
    models: list[str] = []
    for item in data.get("data", []):
        mid = item.get("id")
        if isinstance(mid, str) and mid:
            models.append(mid)
    return models


def _extract_models_from_ollama_tags_payload(data: dict) -> list[str]:
    # Native Ollama payload shape: {"models": [{"name": "..."}]}
    models: list[str] = []
    for item in data.get("models", []):
        name = item.get("name")
        if isinstance(name, str) and name:
            models.append(name)
    return models


def _list_models(client: httpx.Client, base_url: str) -> tuple[list[str], str]:
    """Discover available models from Ollama Cloud.

    Tries multiple endpoints because account/API deployments differ:
    1) OpenAI-compatible /models
    2) Native Ollama /api/tags
    """
    errors: list[str] = []

    # 1) OpenAI-compatible endpoint
    try:
        r = client.get(f"{base_url}/models")
        if r.status_code < 400:
            data = r.json()
            models = _extract_models_from_openai_payload(data)
            if models:
                return sorted(set(models)), "openai:/models"
            errors.append("openai:/models returned no model ids")
        else:
            errors.append(f"openai:/models HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"openai:/models {type(exc).__name__}: {exc}")

    # 2) Native Ollama endpoint
    try:
        r = client.get(f"{base_url}/api/tags")
        if r.status_code < 400:
            data = r.json()
            models = _extract_models_from_ollama_tags_payload(data)
            if models:
                return sorted(set(models)), "ollama:/api/tags"
            errors.append("ollama:/api/tags returned no model names")
        else:
            errors.append(f"ollama:/api/tags HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ollama:/api/tags {type(exc).__name__}: {exc}")

    raise RuntimeError("Model discovery failed: " + " | ".join(errors))


def _bench_once(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RunSample:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    try:
        with client.stream("POST", url, json=payload) as resp:
            status_code = resp.status_code
            first_byte_time: float | None = None
            chunks: list[str] = []

            for chunk in resp.iter_text():
                if first_byte_time is None:
                    first_byte_time = (time.perf_counter() - start) * 1000.0
                if chunk:
                    chunks.append(chunk)

            total_ms = (time.perf_counter() - start) * 1000.0
            body = "".join(chunks)

            if status_code >= 400:
                return RunSample(
                    model=model,
                    run_idx=-1,
                    ok=False,
                    status_code=status_code,
                    ttfb_ms=first_byte_time,
                    total_ms=total_ms,
                    response_chars=len(body),
                    error=_short_err(body or f"HTTP {status_code}"),
                )

            return RunSample(
                model=model,
                run_idx=-1,
                ok=True,
                status_code=status_code,
                ttfb_ms=first_byte_time,
                total_ms=total_ms,
                response_chars=len(body),
                error="",
            )
    except Exception as exc:  # noqa: BLE001
        total_ms = (time.perf_counter() - start) * 1000.0
        return RunSample(
            model=model,
            run_idx=-1,
            ok=False,
            status_code=None,
            ttfb_ms=None,
            total_ms=total_ms,
            response_chars=0,
            error=_short_err(f"{type(exc).__name__}: {exc}"),
        )


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values]
    if not vals:
        return None
    return statistics.mean(vals)


def _p95(values: Iterable[float]) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    # nearest-rank p95
    idx = max(0, min(len(vals) - 1, int(round(0.95 * len(vals) + 0.5)) - 1))
    return vals[idx]


def main() -> int:
    _ensure_stdout_utf8()
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Benchmark latency for Ollama Cloud models."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        help="Ollama OpenAI-compatible base URL (default: https://ollama.com/v1)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OLLAMA_API_KEY",
        help="Environment variable name for Ollama API key.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of benchmark requests per model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Max tokens for each benchmark request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for each benchmark request.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Benchmark only this model id (repeatable).",
    )
    parser.add_argument(
        "--include-non-cloud",
        action="store_true",
        help="Also benchmark model ids that do not include 'cloud'.",
    )
    parser.add_argument(
        "--report-prefix",
        default="ollama_cloud_latency",
        help="Prefix for report files under reports/.",
    )
    parser.add_argument(
        "--prompt-preset",
        choices=sorted(PROMPT_PRESETS.keys()),
        default="tiny",
        help="Prompt shape for latency realism. tiny|ult_like|cweval_like",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Custom prompt string (overrides --prompt-preset).",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        print(f"Missing API key in env var: {args.api_key_env}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(args.timeout)

    with httpx.Client(headers=headers, timeout=timeout) as client:
        try:
            discovered, source = _list_models(client, base_url)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to list models: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        selected = list(args.model) if args.model else discovered
        if not args.include_non_cloud:
            cloud_only = [m for m in selected if "cloud" in m.lower()]
            # If discovery returns non-cloud names only, don't force a hard
            # failure. Benchmark what the API actually exposed.
            if cloud_only:
                selected = cloud_only

        selected = sorted(set(selected))
        if not selected:
            print("No models selected after filtering.", file=sys.stderr)
            return 1

        print(f"Discovered models: {len(discovered)} (source={source})")
        print(f"Selected models : {len(selected)}")
        print(f"Runs/model      : {args.runs}")
        prompt = args.prompt if args.prompt else PROMPT_PRESETS[args.prompt_preset]
        print(f"Prompt preset   : {args.prompt_preset}" if not args.prompt else "Prompt preset   : custom")
        print(f"Prompt chars    : {len(prompt)}")
        print("")

        samples: list[RunSample] = []
        for model in selected:
            print(f"[model] {model}")
            for i in range(1, args.runs + 1):
                s = _bench_once(
                    client=client,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                s.run_idx = i
                samples.append(s)
                status = "ok" if s.ok else "fail"
                print(
                    f"  run {i}/{args.runs}: {status} "
                    f"status={s.status_code or '-'} "
                    f"ttfb={_fmt_ms(s.ttfb_ms)}ms total={_fmt_ms(s.total_ms)}ms"
                )
                if not s.ok and s.error:
                    print(f"    err: {s.error}")
            print("")

    # Aggregate
    by_model: dict[str, list[RunSample]] = {}
    for s in samples:
        by_model.setdefault(s.model, []).append(s)

    rows: list[dict[str, str]] = []
    for model in sorted(by_model):
        group = by_model[model]
        oks = [s for s in group if s.ok]
        ttfb_vals = [s.ttfb_ms for s in oks if s.ttfb_ms is not None]
        total_vals = [s.total_ms for s in oks if s.total_ms is not None]
        err_msgs = [s.error for s in group if (not s.ok and s.error)]
        row = {
            "model": model,
            "runs": str(len(group)),
            "ok_runs": str(len(oks)),
            "success_rate_pct": f"{(len(oks) / len(group) * 100.0):.1f}",
            "ttfb_mean_ms": _fmt_ms(_mean(ttfb_vals)),
            "ttfb_p95_ms": _fmt_ms(_p95(ttfb_vals)),
            "total_mean_ms": _fmt_ms(_mean(total_vals)),
            "total_p95_ms": _fmt_ms(_p95(total_vals)),
            "sample_error": _short_err(err_msgs[0] if err_msgs else ""),
        }
        rows.append(row)

    # Write reports
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"{args.report_prefix}_{ts}.csv"
    md_path = reports_dir / f"{args.report_prefix}_{ts}.md"

    fields = [
        "model",
        "runs",
        "ok_runs",
        "success_rate_pct",
        "ttfb_mean_ms",
        "ttfb_p95_ms",
        "total_mean_ms",
        "total_p95_ms",
        "sample_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ollama Cloud Latency Benchmark",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Base URL: `{base_url}`",
        f"- Runs/model: `{args.runs}`",
        f"- Timeout(s): `{args.timeout}`",
        f"- Max tokens: `{args.max_tokens}`",
        f"- Prompt preset: `{args.prompt_preset if not args.prompt else 'custom'}`",
        f"- Prompt chars: `{len(prompt)}`",
        "",
        "| model | runs | ok | success % | mean ttfb (ms) | p95 ttfb (ms) | mean total (ms) | p95 total (ms) | sample error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['runs']} | {r['ok_runs']} | {r['success_rate_pct']} | "
            f"{r['ttfb_mean_ms']} | {r['ttfb_p95_ms']} | {r['total_mean_ms']} | {r['total_p95_ms']} | "
            f"{r['sample_error'] or '-'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Done.")
    print(f"CSV report: {csv_path}")
    print(f"MD report : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
