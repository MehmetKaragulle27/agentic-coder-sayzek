"""Aggregate metrics across N replicate runs into mean +/- std.

Usage:
    python scripts/aggregate_replicates.py ult \
        eval_results_paper_fold1 \
        eval_results_paper_fold2 \
        eval_results_paper_fold3

Produces a markdown table with mean +/- std for each metric across the
provided replicate directories. Rate-limited cases are *excluded* from
each replicate before aggregation (they aren't real pipeline outcomes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev


_RL_HINTS = ("429", "rate_limit", "rate limit", "tokens per day", "per minute")


def _is_rl(err: str | None) -> bool:
    e = (err or "").lower()
    return bool(e) and any(h in e for h in _RL_HINTS)


def _load_replicate(bench: str, root: Path) -> dict | None:
    d = root / bench
    files = [f for f in d.glob("*.json") if f.name not in ("summary.json", "summary.md")]
    if not files:
        return None

    passed = failed = 0
    tests_run = tests_passed = 0
    iters: list[float] = []
    times: list[float] = []
    covs: list[float] = []

    for f in files:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_rl(j.get("error")):
            continue
        if j.get("passed"):
            passed += 1
        else:
            failed += 1
        tests_run += j.get("tests_run", 0) or 0
        tests_passed += j.get("tests_passed", 0) or 0
        iters.append(j.get("iterations", 0) or 0)
        times.append(j.get("elapsed_seconds", 0) or 0)
        if j.get("coverage") is not None:
            covs.append(j["coverage"])

    n = passed + failed
    if n == 0:
        return None
    return {
        "n": n,
        "case_pass": passed / n,
        "test_pass": (tests_passed / tests_run) if tests_run else 0.0,
        "avg_coverage": mean(covs) if covs else 0.0,
        "avg_iters": mean(iters) if iters else 0.0,
        "avg_time": mean(times) if times else 0.0,
    }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), pstdev(vals)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: aggregate_replicates.py <benchmark> <run_dir1> <run_dir2> [...]"
        )
        sys.exit(1)

    bench = sys.argv[1]
    roots = [Path(p) for p in sys.argv[2:]]

    reps = []
    for r in roots:
        stats = _load_replicate(bench, r)
        if stats is None:
            print(f"[warn] no data for {bench} in {r}")
            continue
        reps.append((r, stats))

    if not reps:
        print("no replicates loaded")
        sys.exit(1)

    print(f"\n## {bench} -- {len(reps)} replicate(s)\n")
    print("| Replicate | N | Case pass | Test pass | Coverage | Iters | Time (s) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r, s in reps:
        print(
            f"| {r.name} | {s['n']} | {s['case_pass']*100:.1f}% | "
            f"{s['test_pass']*100:.1f}% | {s['avg_coverage']:.1f}% | "
            f"{s['avg_iters']:.2f} | {s['avg_time']:.1f} |"
        )

    def col(key: str) -> list[float]:
        return [s[key] for _, s in reps]

    cp_m, cp_s = _mean_std(col("case_pass"))
    tp_m, tp_s = _mean_std(col("test_pass"))
    cv_m, cv_s = _mean_std(col("avg_coverage"))
    it_m, it_s = _mean_std(col("avg_iters"))
    tm_m, tm_s = _mean_std(col("avg_time"))

    print()
    print("**Mean +/- std across replicates**")
    print()
    print(f"- Case pass rate : {cp_m*100:.1f}% +/- {cp_s*100:.1f}%")
    print(f"- Test pass rate : {tp_m*100:.1f}% +/- {tp_s*100:.1f}%")
    print(f"- Avg coverage   : {cv_m:.1f}% +/- {cv_s:.1f}%")
    print(f"- Avg iterations : {it_m:.2f} +/- {it_s:.2f}")
    print(f"- Avg time (s)   : {tm_m:.1f} +/- {tm_s:.1f}")


if __name__ == "__main__":
    main()
