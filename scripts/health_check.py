"""Cross-benchmark health check for an evaluation results directory."""

import json
import sys
from pathlib import Path


def main(root_dir: str = "eval_results_phase6_v2") -> None:
    root = Path(root_dir)
    if not root.exists():
        print(f"no such directory: {root}")
        return

    print(
        f"{'benchmark':20} {'cases':>6} {'errored':>8} {'zero-tests':>11} "
        f"{'cases-pass':>11} {'test-pass':>10} {'coverage':>10}"
    )
    print("-" * 90)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = [
            f for f in d.glob("*.json")
            if f.name not in ("summary.json", "comparison.json")
        ]
        errored = zero = case_pass = 0
        tests_run = tests_pass = 0
        covs = []
        for f in files:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostic tooling
                continue
            err = j.get("error")
            err_type = (j.get("pipeline_state") or {}).get("error_type")
            if err or err_type in ("environment_error", "docker_error"):
                errored += 1
            if j.get("tests_run", 0) == 0 and not err:
                zero += 1
            if j.get("passed"):
                case_pass += 1
            tests_run += j.get("tests_run", 0)
            tests_pass += j.get("tests_passed", 0)
            cov = j.get("coverage")
            if cov is not None:
                covs.append(cov)

        tp = (tests_pass / tests_run * 100) if tests_run else 0.0
        cp = (case_pass / len(files) * 100) if files else 0.0
        cov_mean = (sum(covs) / len(covs)) if covs else None
        cov_str = f"{cov_mean:.1f}%" if cov_mean is not None else "-"
        print(
            f"{d.name:20} {len(files):>6} {errored:>8} {zero:>11} "
            f"{cp:>10.1f}% {tp:>9.1f}% {cov_str:>10}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2")
