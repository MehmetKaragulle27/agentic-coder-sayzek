"""Merge two or more benchmark runs of the same dataset.

For each case_id we keep the *best* clean run (non-rate-limited) we can
find. If every run for a given case is rate-limited, we keep one as a
representative so the total count is preserved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


_RL_HINTS = ("429", "rate_limit", "rate limit", "tokens per day", "per minute")


def _is_rl(case: dict) -> bool:
    e = (case.get("error") or "").lower()
    return bool(e) and any(h in e for h in _RL_HINTS)


def _load_cases(d: Path):
    out = {}
    for f in d.glob("*.json"):
        if f.name in ("summary.json", "summary.md", "comparison.json"):
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "case_id" in j:
            out[j["case_id"]] = j
    return out


def main(dirs: list[str]) -> None:
    runs = [(Path(d), _load_cases(Path(d))) for d in dirs]
    all_ids = set()
    for _, m in runs:
        all_ids.update(m.keys())

    merged = []
    tally = {"clean_pref": 0, "rl_fallback": 0, "missing": 0}
    per_run_source = {str(p): 0 for p, _ in runs}

    for cid in sorted(all_ids):
        chosen = None
        for p, m in runs:
            case = m.get(cid)
            if case and not _is_rl(case):
                chosen = (p, case)
                tally["clean_pref"] += 1
                break
        if chosen is None:
            for p, m in runs:
                case = m.get(cid)
                if case:
                    chosen = (p, case)
                    tally["rl_fallback"] += 1
                    break
        if chosen is None:
            tally["missing"] += 1
            continue
        per_run_source[str(chosen[0])] += 1
        merged.append(chosen[1])

    # Recompute summary over merged set, excluding rate-limited.
    clean = [c for c in merged if not _is_rl(c)]
    total = len(merged)
    passed = sum(1 for c in clean if c.get("passed"))
    tests_run = sum(c.get("tests_run", 0) for c in clean)
    tests_passed = sum(c.get("tests_passed", 0) for c in clean)
    cov_vals = [c["coverage"] for c in clean if c.get("coverage") is not None]
    iter_vals = [c.get("iterations", 0) for c in clean]

    print(f"Merged {total} cases across {len(runs)} runs.")
    print(f"  clean picks  : {tally['clean_pref']}")
    print(f"  rl fallbacks : {tally['rl_fallback']}")
    print(f"  missing      : {tally['missing']}")
    print()
    print("Source breakdown:")
    for p, n in per_run_source.items():
        print(f"  {n:>3}  {p}")
    print()
    print("Aggregate on CLEAN merged subset:")
    print(f"  total clean cases : {len(clean)}")
    print(f"  pass rate (case)  : {passed}/{len(clean)}  "
          f"({passed/max(len(clean),1):.1%})")
    print(f"  tests passed      : {tests_passed}/{tests_run}  "
          f"({tests_passed/max(tests_run,1):.1%})")
    if cov_vals:
        print(f"  avg coverage      : {mean(cov_vals):.1f}%")
    print(f"  avg iterations    : {mean(iter_vals):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or [
        "eval_results_paper/cweval",
        "eval_results_paper/cweval/eval_results_paper/cweval",
    ])
