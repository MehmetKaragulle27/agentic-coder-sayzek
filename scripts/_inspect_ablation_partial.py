"""Quick inspection of in-progress ablation runs."""
import json, pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'eval_results_paper_ablation/ablation/ult')
variants = sorted([d for d in root.iterdir() if d.is_dir()])

print(f"## Ablation in progress -- {len(variants)} variants completed (of 32 expected)\n")
print(f"{'variant':<38} {'N':>3} {'pass':>6} {'tests':>10} {'iter':>5} {'rl':>3}")
print('-' * 70)

rows = []
for v in variants:
    cases = sorted((v / 'ult').glob('ult-*.json')) if (v / 'ult').exists() else sorted(v.glob('ult-*.json'))
    if not cases:
        continue
    passed = 0
    rl = 0
    tests_p = 0
    tests_r = 0
    iters = []
    for p in cases:
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('passed'):
            passed += 1
        err = (d.get('error') or '').lower()
        if any(k in err for k in ['429', 'rate', 'quota', 'timed out', 'timeout']):
            rl += 1
        tests_p += d.get('tests_passed', 0) or 0
        tests_r += d.get('tests_run', 0) or 0
        iters.append(d.get('iterations', 0))

    n = len(cases)
    pass_pct = passed / n * 100 if n else 0
    test_pct = tests_p / tests_r * 100 if tests_r else 0
    avg_it = sum(iters) / len(iters) if iters else 0
    rows.append((v.name, n, pass_pct, tests_p, tests_r, test_pct, avg_it, rl))
    print(f"{v.name:<38} {n:>3} {pass_pct:>5.1f}% {tests_p:>4}/{tests_r:<4} {avg_it:>5.2f} {rl:>3}")

print()
print("== Sorted by case-pass rate ==")
for r in sorted(rows, key=lambda x: -x[2]):
    print(f"  {r[2]:>5.1f}%  test={r[5]:>5.1f}%  iter={r[6]:.2f}  rl={r[7]:>2}  {r[0]}")
