import json, pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'eval_results_paper_fold1/security')
rows = []
for p in sorted(root.glob('sec-*.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    gates = {g['gate_name']: g['passed'] for g in d['gate_results']}
    err = d.get('error') or ''
    rl = any(k in err.lower() for k in ('429', 'rate', 'quota', 'timed out', 'timeout'))
    rows.append((
        d['case_id'],
        d['passed'],
        d['tests_passed'],
        d['tests_run'],
        d['iterations'],
        gates.get('sast'),
        gates.get('dependency'),
        gates.get('sandbox'),
        err,
        rl,
    ))

fmt = '{:<32} {:<6} {:<8} {:<5} {:<6} {:<5} {:<6} {:<4} {}'
print(fmt.format('case', 'pass', 'tests', 'iter', 'sast', 'dep', 'sbox', 'rl', 'err'))
for r in rows:
    case, passed, tp, tr, it, sa, de, sb, err, rl = r
    print(fmt.format(
        case,
        str(passed),
        f'{tp}/{tr}',
        str(it),
        str(sa),
        str(de),
        str(sb),
        str(rl),
        (err[:80] if err else '-'),
    ))
print()
passed_cnt = sum(1 for r in rows if r[1])
rl_cnt = sum(1 for r in rows if r[9])
print(f'summary: {passed_cnt}/{len(rows)} case-pass | rate-limited: {rl_cnt}')

tp = sum(r[2] for r in rows)
tr = sum(r[3] for r in rows)
print(f'test-level: {tp}/{tr} = {100*tp/tr:.1f}%')
