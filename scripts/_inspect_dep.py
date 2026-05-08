import json, pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'eval_results_paper_fold1/dep_hallucination')
rows = []
for p in sorted(root.glob('dep-*.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    gates = {g['gate_name']: g for g in d['gate_results']}
    dep = gates.get('dependency')
    dep_findings = []
    if dep and dep['findings']:
        for f in dep['findings']:
            if f.get('code') in ('PHANTOM-PKG', 'HALLUCINATED-DEP', 'PHANTOM'):
                dep_findings.append(f.get('message', '')[:70])
            else:
                dep_findings.append(f.get('code', '?'))
    err = d.get('error') or ''
    rl = any(k in err.lower() for k in ('429', 'rate', 'quota', 'timed out', 'timeout'))
    rows.append((
        d['case_id'],
        d['passed'],
        f"{d['tests_passed']}/{d['tests_run']}",
        d['iterations'],
        'PASS' if (dep and dep['passed']) else 'FAIL',
        dep_findings,
        err,
        rl,
    ))

print(f"{'case':<22} {'pass':<6} {'tests':<8} {'it':<3} {'dep':<5} {'findings':<40}")
for case, passed, tests, it, dep, findings, err, rl in rows:
    f = (findings[0] if findings else '-')[:60]
    print(f'{case:<22} {str(passed):<6} {tests:<8} {it:<3} {dep:<5} {f}')

print()
print('Benchmark semantics: FAIL on dependency gate = detected phantom = CORRECT for phantom_* cases')
print('PASS on dependency gate = no phantom detected = CORRECT for clean_* cases')
print()

detected = 0
total_phantom = 0
for case, passed, tests, it, dep, findings, err, rl in rows:
    if case.startswith(('dep-phantom',)):
        total_phantom += 1
        if dep == 'FAIL':
            detected += 1

clean_ok = 0
total_clean = 0
for case, passed, tests, it, dep, findings, err, rl in rows:
    if case.startswith('dep-clean'):
        total_clean += 1
        if dep == 'PASS':
            clean_ok += 1

print(f'Phantom detection: {detected}/{total_phantom}')
print(f'Clean-case accepted: {clean_ok}/{total_clean}')
rl_cnt = sum(1 for r in rows if r[7])
print(f'Rate-limit contamination: {rl_cnt}')
