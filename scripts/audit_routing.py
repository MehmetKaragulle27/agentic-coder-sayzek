"""Find cases where the router mis-classified a unit-test job as an explanation."""
from __future__ import annotations

import json
from pathlib import Path


def main(root: str = "eval_results_phase6_v2") -> None:
    root_p = Path(root)
    for bench_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        mis, total = 0, 0
        misrouted_ids = []
        for f in sorted(bench_dir.glob("*.json")):
            if f.name in ("summary.json", "comparison.json"):
                continue
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostic tooling
                continue
            gates = j.get("gate_results") or []
            names = {g.get("gate_name") for g in gates}
            total += 1
            if ({"explanation_judge", "complexity"} & names) and "sandbox" not in names:
                mis += 1
                misrouted_ids.append(j.get("case_id"))
        tag = "OK" if mis == 0 else "MISROUTE"
        print(f"[{tag:8}] {bench_dir.name:20} {mis}/{total} cases misrouted")
        for cid in misrouted_ids:
            print(f"           - {cid}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2")
