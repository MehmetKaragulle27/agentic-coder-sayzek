"""Audit ablation runs for non-pipeline contamination (DNS, rate-limits, etc.)."""
import json, pathlib, re, statistics

ROOT = pathlib.Path('eval_results_paper_ablation/ablation/ult')

NETWORK_HINTS = (
    'getaddrinfo failed',
    'name or service not known',
    'temporary failure in name resolution',
    'connection refused',
    'no route to host',
    'network is unreachable',
)
RATE_LIMIT_HINTS = (
    '429',
    'rate_limit',
    'rate limit',
    'tokens per day',
    'tokens per minute',
    'quota exceeded',
)


def classify_error(err: str) -> str:
    if not err:
        return 'none'
    e = err.lower()
    if any(h in e for h in NETWORK_HINTS):
        return 'network/dns'
    if any(h in e for h in RATE_LIMIT_HINTS):
        return 'rate_limit'
    if 'timed out' in e or 'timeout' in e:
        return 'timeout'
    return 'other'


def variant_stats(folder: pathlib.Path) -> dict:
    name = folder.name
    cases = sorted((folder / 'ult').glob('ult-*.json'))
    if not cases:
        return None
    errors = {'none': 0, 'network/dns': 0, 'rate_limit': 0, 'timeout': 0, 'other': 0}
    passed = 0
    tests_p = tests_r = 0
    iters = []
    for p in cases:
        d = json.loads(p.read_text(encoding='utf-8'))
        cls = classify_error(d.get('error') or '')
        errors[cls] += 1
        if d.get('passed'):
            passed += 1
        tests_p += d.get('tests_passed', 0) or 0
        tests_r += d.get('tests_run', 0) or 0
        iters.append(d.get('iterations', 0))
    n = len(cases)
    contaminated = errors['network/dns'] + errors['rate_limit'] + errors['timeout']
    clean_n = n - contaminated
    clean_pass = passed  # bug-shaped: passed implies clean already
    return {
        'name': name,
        'n': n,
        'passed': passed,
        'errors': errors,
        'contaminated': contaminated,
        'clean_n': clean_n,
        'clean_case_pass': passed / clean_n if clean_n else 0,
        'raw_case_pass': passed / n if n else 0,
        'test_pass': tests_p / tests_r if tests_r else 0,
        'avg_iter': sum(iters) / len(iters) if iters else 0,
    }


variants = [variant_stats(d) for d in sorted(ROOT.iterdir()) if d.is_dir()]
variants = [v for v in variants if v]

# Contamination per variant
print(f"{'variant':<38} {'cases':>6} {'net':>4} {'rl':>3} {'to':>3} {'oth':>4} {'raw%':>6} {'clean%':>7}")
print('-' * 80)
total_contam = 0
for v in variants:
    e = v['errors']
    print(f"{v['name']:<38} {v['n']:>6} {e['network/dns']:>4} {e['rate_limit']:>3} "
          f"{e['timeout']:>3} {e['other']:>4} {v['raw_case_pass']*100:>5.1f}% "
          f"{v['clean_case_pass']*100:>6.1f}%")
    total_contam += v['contaminated']

print(f"\nTotal contaminated cases (DNS+rate-limit+timeout): {total_contam} of {sum(v['n'] for v in variants)}")

# Recompute the headline numbers, excluding contaminated cases
print("\n=== Headline (CLEAN) ===")
bare = next(v for v in variants if v['name'] == 'sast=off_dep=off_judge=off_k=0')
full = next(v for v in variants if v['name'] == 'sast=on_dep=on_judge=on_k=5')
print(f"Bare LLM (no gates, k=0): clean_n={bare['clean_n']} pass={bare['passed']} "
      f"clean_pass={bare['clean_case_pass']*100:.1f}%  (raw {bare['raw_case_pass']*100:.1f}%)")
print(f"Full pipeline (all on, k=5): clean_n={full['clean_n']} pass={full['passed']} "
      f"clean_pass={full['clean_case_pass']*100:.1f}%  (raw {full['raw_case_pass']*100:.1f}%)")

# Recompute marginal effects on CLEAN cases
print("\n=== Marginal axis effects (CLEAN cases only) ===")

def axis_means(axis, val):
    runs = [v['clean_case_pass'] for v in variants
            if f'{axis}={val}' in v['name'] and v['clean_n'] >= 10]
    return statistics.mean(runs) * 100 if runs else 0


for axis in ['sast', 'dep', 'judge']:
    on_m = axis_means(axis, 'on')
    off_m = axis_means(axis, 'off')
    print(f"  {axis:<6} on={on_m:>5.1f}%  off={off_m:>5.1f}%  delta={on_m-off_m:+.1f} pp")

# k effect on clean cases
print("\n=== Repair depth k (CLEAN cases only) ===")
for k in [0, 1, 3, 5]:
    runs = [v['clean_case_pass'] for v in variants
            if f'k={k}' in v['name'] and v['clean_n'] >= 10]
    n_clean = [v['clean_n'] for v in variants if f'k={k}' in v['name']]
    print(f"  k={k}  case={statistics.mean(runs)*100:>5.1f}%  "
          f"clean_n_total={sum(n_clean)}/{8*20}")
