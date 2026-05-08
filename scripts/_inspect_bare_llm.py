"""Investigate why bare-LLM variant shows 0% test-pass."""
import json, pathlib

variant = pathlib.Path('eval_results_paper_ablation/ablation/ult/sast=off_dep=off_judge=off_k=0/ult')
cases = sorted(variant.glob('ult-*.json'))

print(f"Inspecting {len(cases)} cases in bare-LLM variant\n")
print(f"{'case':<35} {'pass':<6} {'tests':<10} {'iter':<5} {'status':<25} {'error_type':<25}")
print('-' * 110)

reasons = {}
for p in cases:
    d = json.loads(p.read_text(encoding='utf-8'))
    s = d.get('pipeline_state', {})
    status = s.get('status', '?')
    err_type = s.get('error_type', '-') or '-'
    err_msg = s.get('error_message', '') or ''

    print(f"{d['case_id']:<35} {str(d['passed']):<6} "
          f"{d['tests_passed']}/{d['tests_run']:<6} "
          f"{d['iterations']:<5} {status:<25} {err_type[:24]:<25}")
    reasons.setdefault(status, 0)
    reasons[status] += 1

print(f"\n=== Status distribution ===")
for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} {v}")

# Now inspect ONE failing case to see what actually happened
print("\n=== Sample raw case (first one) ===")
sample = json.loads(cases[0].read_text(encoding='utf-8'))
print(f"case_id: {sample['case_id']}")
print(f"passed: {sample['passed']}")
print(f"tests_run: {sample['tests_run']}, tests_passed: {sample['tests_passed']}")
print(f"iterations: {sample['iterations']}")
print(f"gates: {[g['gate_name'] + '=' + str(g['passed']) for g in sample.get('gate_results', [])]}")
sb = next((g for g in sample.get('gate_results', []) if g['gate_name'] == 'sandbox'), None)
if sb:
    details = sb.get('details') or ''
    print(f"\n--- Sandbox details (first 1500 chars) ---")
    print(details[:1500])
