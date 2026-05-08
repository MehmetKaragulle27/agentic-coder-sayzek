"""Summarize per-case test pass rates and coverage from a benchmark run."""

import json
import sys
from pathlib import Path


def main(results_dir: str = "eval_results_phase6_v2/projecttest") -> None:
    rd = Path(results_dir)
    total_run = total_pass = 0
    cov_vals = []
    print(f"{'case':40} {'tests':>10} {'pass':>6} {'%':>6} {'cov':>6}")
    print("-" * 75)
    for f in sorted(rd.glob("pt-*.json")):
        d = json.loads(f.read_text())
        tr = d.get("tests_run", 0)
        tp = d.get("tests_passed", 0)
        cov = d.get("coverage")
        pct = (tp / tr * 100) if tr else 0.0
        cov_str = f"{cov:.0f}%" if cov is not None else "-"
        print(f"{d['case_id']:40} {tr:>10} {tp:>6} {pct:>5.1f}% {cov_str:>6}")
        total_run += tr
        total_pass += tp
        if cov is not None:
            cov_vals.append(cov)
    print("-" * 75)
    print(f"TOTAL tests: {total_pass}/{total_run} passed "
          f"({(total_pass / total_run * 100 if total_run else 0):.1f}%)")
    if cov_vals:
        print(f"Mean coverage: {sum(cov_vals) / len(cov_vals):.1f}%  "
              f"(across {len(cov_vals)} cases)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2/projecttest")
