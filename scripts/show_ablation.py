"""Print a quick summary table of an ablation run so far (partial runs OK)."""

import json
import sys
from pathlib import Path


def main(root: str = "eval_results_ablation/ablation/ult") -> None:
    p = Path(root)
    if not p.exists():
        print(f"no such directory: {p}")
        return

    rows = []
    for d in sorted(p.iterdir()):
        if not d.is_dir():
            continue
        candidates = list(d.rglob("summary.json"))
        if not candidates:
            continue
        summary = json.loads(candidates[0].read_text(encoding="utf-8"))
        rows.append((d.name, summary))

    if not rows:
        print("no completed variants yet")
        return

    hdr = f"{'variant':42} {'cases-pass':>10} {'test-pass':>10} {'coverage':>10} {'iters':>7} {'time(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        cp = m.get("pass_rate", 0) * 100
        tp = m.get("test_pass_rate", 0) * 100
        cov = m.get("avg_coverage") or 0.0
        it = m.get("avg_iterations") or 0.0
        tm = m.get("avg_time") or 0.0
        print(f"{name:42} {cp:>9.1f}% {tp:>9.1f}% {cov:>9.1f}% {it:>7.2f} {tm:>8.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_ablation/ablation/ult")
