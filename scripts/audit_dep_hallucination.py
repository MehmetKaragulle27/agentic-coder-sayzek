"""Check whether dep_hallucination cases actually exercise the dep-validation gate."""
from __future__ import annotations

import json
from pathlib import Path


def main(results_root: str = "eval_results_phase6_v2") -> None:
    d = Path(results_root) / "dep_hallucination"
    header = f"{'case':25} {'tests':>6} {'pass':>5} {'cov':>5}  {'dep':>6} {'sast':>6} {'sandbox':>8}"
    print(header)
    print("-" * len(header))
    for f in sorted(d.glob("*.json")):
        if f.name in ("summary.json", "summary.md"):
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        gates = {g["gate_name"]: g["passed"] for g in (j.get("gate_results") or [])}
        cov = j.get("coverage")
        cov_s = f"{cov:.0f}%" if cov is not None else "-"
        print(
            f"{j['case_id']:25} {j.get('tests_run',0):>6} {j.get('tests_passed',0):>5} "
            f"{cov_s:>5}  {str(gates.get('dependency')):>6} {str(gates.get('sast')):>6} "
            f"{str(gates.get('sandbox')):>8}"
        )


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2")
