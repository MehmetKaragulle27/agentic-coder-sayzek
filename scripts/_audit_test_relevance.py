"""Audit test relevance: detect cases where generated tests don't actually
exercise the function under test.

Why this matters
----------------
ULT-style benchmarks let the LLM generate BOTH the code and the tests. If we
only score "tests pass", a model can get 100% by writing a fictional class
(e.g. ``Calculator``) and tests for it -- ignoring the original target
function entirely. This is the classic self-evaluating test generation
failure mode.

This script extracts test names from saved pytest stdout (in
``gate_results[name='sandbox'].details``) and checks whether ANY test name
references the target function from the case_id. Cases with zero references
are flagged as "likely gaming" candidates.

Usage:
    python scripts/_audit_test_relevance.py [<run_or_ablation_root>] [--variant glob]

Examples:
    python scripts/_audit_test_relevance.py eval_results_paper_ablation
    python scripts/_audit_test_relevance.py eval_results_with_relevance
    python scripts/_audit_test_relevance.py eval_results_with_relevance/ult
    python scripts/_audit_test_relevance.py eval_results_paper_ablation \
        --variant "sast=off_dep=off_judge=off_*"
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import re
import sys
from collections import defaultdict


_TEST_NAME_RE = re.compile(r"test_generated\.py::([\w:]+)::([\w\[\]\-]+)")


def _extract_test_names(pytest_stdout: str) -> list[str]:
    """Parse test class::test_name pairs out of a pytest -v stdout block."""
    return [f"{m.group(1)}::{m.group(2)}" for m in _TEST_NAME_RE.finditer(pytest_stdout)]


def _target_keywords(case_id: str) -> list[str]:
    """Pull out function-name-ish keywords from the case_id.

    ULT IDs look like ``ult-30-DetPiece`` or ``ult-15-compute_comment_stats``.
    We accept (a) the raw last segment, (b) lowercased, (c) snake_case-split
    components longer than 2 chars, and (d) camelCase-split components.
    """
    parts = case_id.split('-')
    if len(parts) < 3:
        return []
    raw = parts[-1]
    keywords: set[str] = {raw, raw.lower()}
    for tok in raw.split('_'):
        if len(tok) >= 3:
            keywords.add(tok.lower())
    # CamelCase split
    for tok in re.findall(r'[A-Z][a-z]+|[a-z]+', raw):
        if len(tok) >= 3:
            keywords.add(tok.lower())
    return [k for k in keywords if k]


def _relevance(test_names: list[str], keywords: list[str]) -> tuple[int, int]:
    """Return (matching_tests, total_tests). A test matches if any keyword
    occurs as a substring (case-insensitive) of the test name."""
    if not test_names:
        return 0, 0
    matched = 0
    for name in test_names:
        nl = name.lower()
        if any(kw in nl for kw in keywords):
            matched += 1
    return matched, len(test_names)


def audit_variant(folder: pathlib.Path) -> dict | None:
    """Only ``ult-*.json`` case files; never ``comparison.json`` / ``summary.json``."""
    cases = sorted(folder.glob('ult-*.json'))
    if not cases:
        return None

    total = len(cases)
    suspicious = []
    relevance_pcts = []
    zero_match_passing = 0
    n_with_tests = 0

    for case_path in cases:
        try:
            d = json.loads(case_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        case_id = d.get('case_id', case_path.stem)
        gates = {g['gate_name']: g for g in d.get('gate_results', [])}
        sandbox = gates.get('sandbox')
        if not sandbox or not sandbox.get('details'):
            continue
        names = _extract_test_names(sandbox['details'])
        if not names:
            continue
        n_with_tests += 1
        keywords = _target_keywords(case_id)
        matched, total_n = _relevance(names, keywords)
        pct = matched / total_n if total_n else 0
        relevance_pcts.append(pct)
        if matched == 0 and d.get('passed'):
            zero_match_passing += 1
            suspicious.append({
                'case_id': case_id,
                'tests_run': d.get('tests_run', 0),
                'tests_passed': d.get('tests_passed', 0),
                'coverage': d.get('coverage', 0),
                'sample_test_names': names[:3],
                'target_keywords': keywords[:5],
            })

    return {
        'variant': folder.name,
        'cases_total': total,
        'cases_with_tests': n_with_tests,
        'avg_relevance_pct': sum(relevance_pcts) / len(relevance_pcts) if relevance_pcts else 0.0,
        'zero_match_passing': zero_match_passing,
        'gaming_rate': zero_match_passing / n_with_tests if n_with_tests else 0.0,
        'suspicious': suspicious,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('results_root', nargs='?', default='eval_results_paper_ablation')
    parser.add_argument('--variant', default='*', help='Glob filter on variant folders')
    parser.add_argument('--show-cases', action='store_true', help='Show suspicious case details')
    args = parser.parse_args()

    root = pathlib.Path(args.results_root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    candidates: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()

    def add(p: pathlib.Path, label_hint: str = "") -> None:
        p = p.resolve()
        if p in seen:
            return
        if not list(p.glob('ult-*.json')):
            return
        seen.add(p)
        candidates.append(p)

    # 1) User pointed at ``.../ult`` directly (evaluate output).
    if root.is_dir() and root.name == 'ult':
        add(root)
    elif (root / 'ult').is_dir():
        add(root / 'ult')

    # 2) Ablation layout: ``<out>/ablation/ult/<variant>/ult/*.json``
    abl_ult = root / 'ablation' / 'ult'
    if abl_ult.is_dir():
        for vdir in sorted(abl_ult.iterdir()):
            if not vdir.is_dir():
                continue
            if not fnmatch.fnmatch(vdir.name, args.variant):
                continue
            inner = vdir / 'ult'
            if inner.is_dir():
                add(inner)

    # 3) Legacy / mixed: any ``<something>/ult`` leaf that holds case JSON
    #    (skip ``ablation/ult`` itself — only variant subfolders have cases).
    for ult_dir in root.rglob('ult'):
        if not ult_dir.is_dir():
            continue
        parent = ult_dir.parent
        if parent.name == 'ablation' and ult_dir.name == 'ult':
            continue
        if ult_dir in seen:
            continue
        if not list(ult_dir.glob('ult-*.json')):
            continue
        # Variant-shaped parent: ``.../sast=on_.../ult``
        if '=' in parent.name and fnmatch.fnmatch(parent.name, args.variant):
            add(ult_dir)

    if not candidates:
        print(
            f"No ULT case folders (ult-*.json) found under {root}. "
            f"For ``evaluate`` output use:\n"
            f"  python scripts/_audit_test_relevance.py {root / 'ult'}\n"
            f"or pass the run root if it contains an ``ult/`` subdir.",
            file=sys.stderr,
        )
        return 1

    print(f"\nTest-relevance audit: {root}")
    print(f"{'Variant':<50} {'N':>3} {'Tests':>5} {'AvgRel':>7} {'Gaming':>7}")
    print('-' * 80)
    summaries = []
    for c in sorted(candidates, key=lambda p: p.parent.name if p.name == 'ult' else p.name):
        s = audit_variant(c)
        if not s:
            continue
        # Use parent name for the ult/ subdir layout
        label = c.parent.name if c.name == 'ult' else c.name
        s['variant'] = label
        summaries.append(s)
        print(f"{label:<50} {s['cases_total']:>3} {s['cases_with_tests']:>5} "
              f"{s['avg_relevance_pct']*100:>6.1f}% {s['gaming_rate']*100:>6.1f}%")

    # Roll-up
    if summaries:
        avg_rel = sum(s['avg_relevance_pct'] for s in summaries) / len(summaries)
        avg_gaming = sum(s['gaming_rate'] for s in summaries) / len(summaries)
        print('-' * 80)
        print(f"{'OVERALL':<50} {'':>3} {'':>5} {avg_rel*100:>6.1f}% {avg_gaming*100:>6.1f}%")

    # Aggregate by repair depth k
    by_k: dict[int, list[dict]] = defaultdict(list)
    for s in summaries:
        m = re.search(r'k=(\d+)', s['variant'])
        if m:
            by_k[int(m.group(1))].append(s)
    if by_k:
        print(f"\n{'By repair depth (k)':<50} {'AvgRel':>7} {'Gaming':>7}")
        print('-' * 80)
        for k in sorted(by_k):
            grp = by_k[k]
            avg_rel = sum(x['avg_relevance_pct'] for x in grp) / len(grp)
            avg_gam = sum(x['gaming_rate'] for x in grp) / len(grp)
            print(f"  k={k} (n={len(grp)} variants){'':<27} {avg_rel*100:>6.1f}% {avg_gam*100:>6.1f}%")

    if args.show_cases:
        print("\n=== Suspicious cases (passing without referencing target) ===\n")
        for s in summaries:
            if not s['suspicious']:
                continue
            print(f"--- {s['variant']} ({len(s['suspicious'])} cases) ---")
            for sc in s['suspicious'][:5]:
                print(f"  {sc['case_id']}  (tests={sc['tests_run']} pass={sc['tests_passed']} cov={sc['coverage']}%)")
                print(f"    target_keywords: {sc['target_keywords']}")
                print(f"    sample tests   : {sc['sample_test_names']}")
            print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
