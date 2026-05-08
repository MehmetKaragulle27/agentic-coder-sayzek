"""Per-case breakdown for both sides of the A/B run."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    for d in (
        Path("eval_results_ab_compare/run_a/ult"),
        Path("eval_results_ab_compare/run_b/ult"),
    ):
        print(f"\n--- {d} ---")
        files = sorted(f for f in d.glob("ult-*.json"))
        for f in files:
            j = json.loads(f.read_text(encoding="utf-8"))
            gates = [(g["gate_name"], g["passed"]) for g in (j.get("gate_results") or [])]
            ps = j.get("pipeline_state") or {}
            cid = j.get("case_id", "?")
            print(
                f"{cid:36} pass={str(j['passed']):5} "
                f"tests={j['tests_run']:>4}/{j['tests_passed']:<4} "
                f"iter={j['iterations']} "
                f"cov={str(j.get('coverage')):>6} "
                f"status={ps.get('status')} err={ps.get('error_type')} "
                f"gates={gates}"
            )


if __name__ == "__main__":
    main()
