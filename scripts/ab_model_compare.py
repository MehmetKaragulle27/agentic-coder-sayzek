"""A/B compare two coding models on the same benchmark slice.

Runs the evaluator twice with identical settings except CODING_MODEL,
then prints a compact side-by-side summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_eval(
    benchmark: str,
    n_cases: int,
    output_dir: Path,
    coding_model: str,
    verbose: bool,
) -> None:
    env = os.environ.copy()
    env["CODING_PROVIDER"] = "ollama"
    env["CODING_MODEL"] = coding_model

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "evaluate",
        "-b",
        benchmark,
        "-o",
        str(output_dir),
        "-n",
        str(n_cases),
    ]
    if verbose:
        cmd.append("-v")

    print(f"\n=== Running {benchmark} with CODING_MODEL={coding_model} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _load_summary(run_dir: Path, benchmark: str) -> dict:
    path = run_dir / benchmark / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def _print_compare(model_a: str, a: dict, model_b: str, b: dict) -> None:
    rows = [
        ("Total cases", a.get("total"), b.get("total")),
        ("Passed", a.get("passed"), b.get("passed")),
        ("Failed", a.get("failed"), b.get("failed")),
        ("Errored", a.get("errored"), b.get("errored")),
        ("Case pass rate", _fmt_pct(a.get("pass_rate")), _fmt_pct(b.get("pass_rate"))),
        (
            "Test pass rate",
            _fmt_pct(a.get("test_pass_rate")),
            _fmt_pct(b.get("test_pass_rate")),
        ),
        ("Avg coverage", f"{a.get('avg_coverage', 0):.1f}%", f"{b.get('avg_coverage', 0):.1f}%"),
        ("Avg iterations", f"{a.get('avg_iterations', 0):.2f}", f"{b.get('avg_iterations', 0):.2f}"),
        ("Avg time (s)", f"{a.get('avg_time', 0):.2f}", f"{b.get('avg_time', 0):.2f}"),
    ]

    print("\n=== A/B Comparison ===")
    print(f"{'Metric':22} | {model_a:28} | {model_b:28}")
    print("-" * 86)
    for metric, va, vb in rows:
        print(f"{metric:22} | {str(va):28} | {str(vb):28}")


def main() -> None:
    p = argparse.ArgumentParser(description="A/B compare two coding models")
    p.add_argument("--benchmark", "-b", default="ult", help="Benchmark name (default: ult)")
    p.add_argument("--n", type=int, default=10, help="Number of cases (default: 10)")
    p.add_argument(
        "--model-a",
        default="glm-5.1:cloud",
        help="First coding model (default: glm-5.1:cloud)",
    )
    p.add_argument(
        "--model-b",
        default="qwen3-coder-next:cloud",
        help="Second coding model (default: qwen3-coder-next:cloud)",
    )
    p.add_argument(
        "--out-root",
        default="eval_results_ab_compare",
        help="Output root directory",
    )
    p.add_argument(
        "--skip-a",
        action="store_true",
        help="Skip Run A (reuse existing summary in out-root/run_a/<bench>/summary.json)",
    )
    p.add_argument(
        "--skip-b",
        action="store_true",
        help="Skip Run B (reuse existing summary in out-root/run_b/<bench>/summary.json)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    run_a = out_root / "run_a"
    run_b = out_root / "run_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    if args.skip_a:
        print(f"Skipping Run A -- reusing {run_a / args.benchmark}")
    else:
        _run_eval(args.benchmark, args.n, run_a, args.model_a, args.verbose)

    if args.skip_b:
        print(f"Skipping Run B -- reusing {run_b / args.benchmark}")
    else:
        _run_eval(args.benchmark, args.n, run_b, args.model_b, args.verbose)

    a = _load_summary(run_a, args.benchmark)
    b = _load_summary(run_b, args.benchmark)
    _print_compare(args.model_a, a, args.model_b, b)

    print("\nDone.")
    print(f"- Run A results: {run_a / args.benchmark}")
    print(f"- Run B results: {run_b / args.benchmark}")


if __name__ == "__main__":
    main()

