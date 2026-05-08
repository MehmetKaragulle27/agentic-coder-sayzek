"""Full audit: scan every saved result for contamination signals.

Produces:
1. A list of contaminated cases per variant/fold/benchmark
2. Rerun recommendations grouped by severity
3. A clean-data sanity reference for what's already usable
"""
import json, pathlib, re
from collections import defaultdict


NETWORK_HINTS = (
    'getaddrinfo failed',
    'name or service not known',
    'temporary failure in name resolution',
    'connection refused',
    'no route to host',
    'network is unreachable',
    'remoteprotocolerror',
    'remote protocol error',
    'connection reset',
    'server disconnected',
)
RATELIMIT_HINTS = (
    '429',
    'rate_limit',
    'rate limit',
    'tokens per day',
    'tokens per minute',
    'quota exceeded',
    'resource_exhausted',
)
TIMEOUT_HINTS = (
    'timed out',
    'timeout',
    'read operation timed out',
)


def classify(err: str) -> str:
    """Return 'clean', 'network', 'rate_limit', 'timeout', or 'other'."""
    if not err:
        return 'clean'
    e = err.lower()
    if any(h in e for h in NETWORK_HINTS):
        return 'network'
    if any(h in e for h in RATELIMIT_HINTS):
        return 'rate_limit'
    if any(h in e for h in TIMEOUT_HINTS):
        return 'timeout'
    return 'other'


def is_empty_pipeline(case_data: dict) -> bool:
    """A case where the pipeline returned no gate results AND no error,
    or where gate_results is empty AND there's an infrastructure error."""
    gates = case_data.get('gate_results') or []
    err = case_data.get('error') or ''
    if not gates and err:
        return True
    return False


def audit_folder(folder: pathlib.Path, label: str) -> dict:
    """Audit every <case>.json in a folder (excluding summary.json)."""
    cases = [p for p in folder.glob('*.json') if p.name != 'summary.json']
    if not cases:
        return None

    counts = {'clean': 0, 'network': 0, 'rate_limit': 0, 'timeout': 0, 'other': 0}
    bad_cases = []

    for p in cases:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue

        err = d.get('error') or ''
        cls = classify(err)
        counts[cls] += 1
        if cls != 'clean':
            bad_cases.append({
                'case_id': d.get('case_id', p.stem),
                'class': cls,
                'error': err[:120],
                'tests_passed': d.get('tests_passed', 0),
                'tests_run': d.get('tests_run', 0),
                'iterations': d.get('iterations', 0),
            })

    n = len(cases)
    bad = n - counts['clean']
    return {
        'label': label,
        'folder': str(folder),
        'n': n,
        'clean': counts['clean'],
        'bad': bad,
        'pct_contam': bad / n if n else 0,
        'counts': counts,
        'bad_cases': bad_cases,
    }


# ─── Scan everything ───────────────────────────────────────────────────
ROOT = pathlib.Path('.')

# 1) Fold benchmarks
FOLDS = ['eval_results_paper_fold1', 'eval_results_paper_fold2', 'eval_results_paper_fold3']
BENCHMARKS = ['security', 'dep_hallucination', 'ult', 'cweval', 'codejudgebench', 'projecttest']

fold_audits = []
for fold in FOLDS:
    for bench in BENCHMARKS:
        folder = ROOT / fold / bench
        if folder.exists():
            a = audit_folder(folder, f'{fold}/{bench}')
            if a:
                fold_audits.append(a)

# 2) Ablation variants
ABLATION_ROOT = ROOT / 'eval_results_paper_ablation/ablation/ult'
ablation_audits = []
if ABLATION_ROOT.exists():
    for v in sorted(ABLATION_ROOT.iterdir()):
        if v.is_dir():
            ult_folder = v / 'ult'
            if ult_folder.exists():
                a = audit_folder(ult_folder, f'ablation/{v.name}')
                if a:
                    ablation_audits.append(a)

# 3) Cross-model (if any)
crossmodel_root = ROOT / 'eval_results_paper_crossmodel'
crossmodel_audits = []
if crossmodel_root.exists():
    for model_dir in sorted(crossmodel_root.iterdir()):
        if model_dir.is_dir():
            for bench_dir in sorted(model_dir.iterdir()):
                if bench_dir.is_dir():
                    a = audit_folder(bench_dir, f'crossmodel/{model_dir.name}/{bench_dir.name}')
                    if a:
                        crossmodel_audits.append(a)


# ─── Print report ──────────────────────────────────────────────────────

def print_section(title, audits, threshold_severe=0.5, threshold_minor=0.0):
    print(f"\n{'=' * 78}")
    print(f"{title}")
    print('=' * 78)
    if not audits:
        print("  (no data)")
        return

    # Header
    print(f"{'location':<55} {'N':>4} {'clean':>5} {'net':>4} {'rl':>3} {'to':>3} "
          f"{'oth':>4} {'%':>5}")
    print('-' * 90)
    for a in audits:
        c = a['counts']
        contam_pct = a['pct_contam'] * 100
        flag = ''
        if a['bad'] > 0 and contam_pct >= threshold_severe * 100:
            flag = '  *** RERUN'
        elif a['bad'] > 0:
            flag = '  rerun?'
        print(f"{a['label']:<55} {a['n']:>4} {a['clean']:>5} "
              f"{c['network']:>4} {c['rate_limit']:>3} {c['timeout']:>3} "
              f"{c['other']:>4} {contam_pct:>4.1f}%{flag}")


print_section("Section 1 — Benchmark folds (eval_results_paper_fold1/2/3)", fold_audits)
print_section("Section 2 — Ablation variants (eval_results_paper_ablation/ablation/ult/*)", ablation_audits)
print_section("Section 3 — Cross-model (eval_results_paper_crossmodel/*)", crossmodel_audits)


# ─── Severity summary + rerun list ─────────────────────────────────────
print("\n" + "=" * 78)
print("RERUN RECOMMENDATIONS (sorted by severity)")
print("=" * 78)

all_audits = fold_audits + ablation_audits + crossmodel_audits

# Severity 1: 50%+ contaminated (must rerun)
must_rerun = [a for a in all_audits if a['pct_contam'] >= 0.5]
if must_rerun:
    print(f"\n[CRITICAL] {len(must_rerun)} runs are >= 50% contaminated -- MUST rerun:")
    for a in must_rerun:
        print(f"  - {a['label']:<55} ({a['bad']}/{a['n']} bad, {a['pct_contam']*100:.0f}%)")

# Severity 2: 20-50%
should_rerun = [a for a in all_audits if 0.2 <= a['pct_contam'] < 0.5]
if should_rerun:
    print(f"\n[HIGH] {len(should_rerun)} runs are 20-50% contaminated -- SHOULD rerun:")
    for a in should_rerun:
        print(f"  - {a['label']:<55} ({a['bad']}/{a['n']} bad, {a['pct_contam']*100:.0f}%)")

# Severity 3: 1-20% (case-level rerun, not whole benchmark)
case_rerun = [a for a in all_audits if 0 < a['pct_contam'] < 0.2]
if case_rerun:
    print(f"\n[LOW] {len(case_rerun)} runs have 1-20% contamination -- specific cases to rerun:")
    for a in case_rerun:
        print(f"\n  {a['label']} ({a['bad']}/{a['n']} bad, {a['pct_contam']*100:.0f}%)")
        for bc in a['bad_cases']:
            print(f"      - {bc['case_id']:<35} [{bc['class']}] {bc['error'][:60]}")

# Clean
clean = [a for a in all_audits if a['pct_contam'] == 0]
print(f"\n[CLEAN] {len(clean)} of {len(all_audits)} runs have ZERO contamination.")

# ─── Generate per-benchmark rerun commands ─────────────────────────────
print("\n" + "=" * 78)
print("SUGGESTED RERUN COMMANDS")
print("=" * 78)

# For folds: if any benchmark in a fold has contamination, rerun that whole benchmark
fold_reruns_by_severity = defaultdict(list)
for a in fold_audits:
    if a['pct_contam'] > 0:
        m = re.match(r'(eval_results_paper_fold\d)/(\w+)', a['label'])
        if m:
            fold = m.group(1)
            bench = m.group(2)
            sev = 'critical' if a['pct_contam'] >= 0.5 else ('high' if a['pct_contam'] >= 0.2 else 'low')
            fold_reruns_by_severity[(fold, sev)].append((bench, a['bad'], a['n']))

if fold_reruns_by_severity:
    print("\n# Rerun fold benchmarks (only the affected benchmarks)")
    for (fold, sev), bench_list in sorted(fold_reruns_by_severity.items()):
        print(f"\n# --- {fold} [{sev}] ---")
        for bench, bad, n in bench_list:
            n_arg = ''
            if bench == 'ult':
                n_arg = '-n 50'
            elif bench == 'cweval':
                n_arg = '-n 50'
            elif bench == 'codejudgebench':
                n_arg = '-n 50'
            elif bench == 'projecttest':
                n_arg = '-n 20'
            print(f"# {bench}: {bad}/{n} contaminated")
            print(f"python -m src.main evaluate -b {bench} -o {fold} {n_arg} -v")

# For ablation: list specific variants that need rerunning
ablation_critical = [a for a in ablation_audits if a['pct_contam'] >= 0.5]
ablation_high = [a for a in ablation_audits if 0.2 <= a['pct_contam'] < 0.5]

if ablation_critical or ablation_high:
    print("\n# --- Ablation rerun ---")
    print("# The full ablation runner doesn't expose a per-variant CLI, so the")
    print("# fastest path is a targeted single-variant rerun via direct env override.")
    print("# Set the variant gates inline via env, then run -b ult -n 20.")
    print("#")
    print("# Variants needing rerun:")
    for a in ablation_critical + ablation_high:
        m = re.match(r'ablation/sast=(on|off)_dep=(on|off)_judge=(on|off)_k=(\d+)', a['label'])
        if m:
            sast, dep, judge, k = m.groups()
            sev = 'CRITICAL' if a['pct_contam'] >= 0.5 else 'HIGH'
            print(f"#   [{sev}] {a['label']:<55} ({a['bad']}/{a['n']} bad, {a['pct_contam']*100:.0f}%)")

print()
