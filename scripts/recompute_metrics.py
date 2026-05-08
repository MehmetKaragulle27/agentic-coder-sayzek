"""Recompute aggregate metrics from saved per-case EvalResult JSONs.

Useful when you've changed metric definitions in ``EvalMetrics`` and want
to see the new numbers for an old run without paying the LLM bill twice.

Note: per-case files written *before* the sandbox-pass-count fix will
still report ``tests_passed=0`` for the broken cases (we can't recover
that information from the raw pytest details string post-hoc unless we
parse it). For an apples-to-apples paper number, re-run the benchmark.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.evaluation.models import EvalMetrics, EvalResult


_PYTEST_SUMMARY = re.compile(
    r"(?P<failed>\d+)\s+failed.*?(?P<passed>\d+)\s+passed"
    r"|(?P<passed_only>\d+)\s+passed",
    re.IGNORECASE,
)


def _backfill_from_details(result: EvalResult) -> EvalResult:
    """Best-effort: parse pytest's '=== N failed, M passed ===' line.

    Old per-case files report ``tests_passed=0`` whenever the case
    overall was marked failed, even if 28/34 sub-tests passed. We try
    to recover the real counts from the sandbox gate's stored details
    string; if that fails we leave the values alone.
    """
    if result.tests_passed > 0:
        return result
    for g in result.gate_results:
        if g.get("gate_name") != "sandbox":
            continue
        details = g.get("details") or ""
        # Walk every "X failed, Y passed" / "Y passed" match and keep
        # the largest numbers we see (pytest can print partial lines).
        best_passed = 0
        best_failed = 0
        for m in _PYTEST_SUMMARY.finditer(details):
            p = int(m.group("passed") or m.group("passed_only") or 0)
            f = int(m.group("failed") or 0)
            if p + f > best_passed + best_failed:
                best_passed, best_failed = p, f
        if best_passed + best_failed > 0:
            result = result.model_copy(
                update={
                    "tests_passed": best_passed,
                    "tests_run": best_passed + best_failed,
                }
            )
        break
    return result


def _is_rate_limited(result: EvalResult) -> bool:
    """Return True when a per-case JSON is actually a rate-limit error.

    Rate-limit errors appear as top-level ``error`` strings (often with
    ``Error code: 429``). Counting them as regular failures inflates
    the failure rate and distorts ablation comparisons.
    """
    msg = (result.error or "").lower()
    return bool(msg) and (
        "429" in msg
        or "rate_limit" in msg
        or "rate limit" in msg
        or "tokens per day" in msg
        or "per minute" in msg
    )


def main(results_dir: str) -> None:
    rd = Path(results_dir)
    files = sorted(p for p in rd.glob("*.json") if p.name != "summary.json")
    if not files:
        print(f"no per-case JSONs found in {rd}")
        return

    all_results: list[EvalResult] = []
    rate_limited = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "case_id" not in data:
                continue
            all_results.append(_backfill_from_details(EvalResult(**data)))
        except Exception as exc:  # noqa: BLE001 - best-effort tooling
            print(f"skip {f.name}: {exc}")

    clean_results = []
    for r in all_results:
        if _is_rate_limited(r):
            rate_limited += 1
        else:
            clean_results.append(r)

    metrics = EvalMetrics.from_results(clean_results, dataset_name=rd.name)
    print(metrics.to_markdown())
    print()
    print(
        f"(recomputed from {len(clean_results)} clean case files in {rd}; "
        f"{rate_limited} rate-limited cases excluded out of {len(all_results)} total)"
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2/projecttest"
    main(target)
