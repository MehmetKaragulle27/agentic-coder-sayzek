"""Gate: Test-relevance validation.

Self-evaluating test generation has a well-known failure mode: the LLM can
satisfy the "all tests pass" criterion by writing tests for a *fictional*
class/function instead of the actual function under test, then writing the
fictional implementation to make those tests pass. The sandbox happily
reports 100% pass because the (fake) tests pass against the (fake) code.

We measured this on ULT: at k=5 with no other gates, ~33% of "passing"
cases had zero tests that referenced the target function name.

This validator catches that. Given the source code (or a target function
name) and the generated test code, it checks:

  1. Does the test code import from ``source_module`` (or whichever
     module the pipeline placed the source under)?
  2. Does at least one test name reference the target function?
  3. If the test file *re-defines* a class or function with the same name
     as the import target, that's a stronger gaming signal -- flag it.

It is intentionally conservative: many true-positive tests share a
function name and import the source module. Only when *none* of these
signals is present do we fail the gate.
"""
from __future__ import annotations

import ast
import re
from typing import Iterable, Optional

from .models import GateResult, Finding, Severity


_DEFAULT_SOURCE_MODULE = "source_module"


def _camel_split(name: str) -> list[str]:
    """Split CamelCase / snake_case into lowercased word tokens of len>=3."""
    tokens: set[str] = set()
    for tok in re.split(r"[_\W]+", name):
        if len(tok) >= 3:
            tokens.add(tok.lower())
    for tok in re.findall(r"[A-Z][a-z]+|[a-z]+", name):
        if len(tok) >= 3:
            tokens.add(tok.lower())
    return list(tokens)


def _candidate_keywords(target: str) -> list[str]:
    base = {target, target.lower()}
    base.update(_camel_split(target))
    return [k for k in base if k]


def _imports_source_module(tree: ast.AST, source_module: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == source_module or alias.name.startswith(f"{source_module}."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == source_module or mod.startswith(f"{source_module}."):
                return True
            # ``from . import source_module`` style
            for alias in node.names:
                if alias.name == source_module:
                    return True
    return False


def _imported_target_names(tree: ast.AST, source_module: str) -> set[str]:
    """Names brought in via ``from source_module import X``."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == source_module or mod.startswith(f"{source_module}."):
                for alias in node.names:
                    out.add(alias.asname or alias.name)
    return out


def _test_function_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                for inner in node.body:
                    if (
                        isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and inner.name.startswith("test_")
                    ):
                        names.append(f"{node.name}.{inner.name}")
    return names


def _redefined_target_names(tree: ast.AST, target: str) -> bool:
    """Returns True if the test file *also* defines a top-level
    function/class with the exact target name -- a strong gaming signal."""
    target_lower = target.lower()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target or node.name.lower() == target_lower:
                return True
    return False


class RelevanceValidator:
    """Detects tests that don't actually exercise the function under test.

    Args:
        source_module: The Python module name the pipeline writes the
            source-under-test to (default ``source_module``).
        min_relevance_signals: How many of the three checks must succeed
            for the gate to pass. Default 1 (any signal is enough).
    """

    def __init__(
        self,
        source_module: str = _DEFAULT_SOURCE_MODULE,
        min_relevance_signals: int = 1,
    ) -> None:
        self.source_module = source_module
        self.min_relevance_signals = max(1, int(min_relevance_signals))

    def validate(
        self,
        test_code: str,
        target_function: Optional[str] = None,
        *,
        source_code: Optional[str] = None,
    ) -> GateResult:
        """Evaluate test relevance to the source under test.

        Args:
            test_code: The generated test file content.
            target_function: Name of the function that should be exercised.
                If None, only the import-from-source check is performed.
            source_code: Optional source code; used to auto-detect the
                target function name(s) when ``target_function`` isn't
                provided (we pick the first top-level ``def`` we find).
        """
        if not test_code or not test_code.strip():
            return GateResult(
                gate_name="relevance",
                passed=False,
                findings=[Finding(
                    severity=Severity.ERROR,
                    code="empty_tests",
                    message="No test code was generated.",
                )],
                details="empty test_code",
            )

        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            return GateResult(
                gate_name="relevance",
                passed=False,
                findings=[Finding(
                    severity=Severity.ERROR,
                    code="test_syntax_error",
                    message=f"Cannot parse test code: {exc.msg}",
                    line=exc.lineno,
                )],
                details=str(exc),
            )

        # Auto-detect target function from source_code if none provided
        target_candidates: list[str] = []
        if target_function:
            target_candidates.append(target_function)
        elif source_code:
            try:
                src_tree = ast.parse(source_code)
                for n in ast.iter_child_nodes(src_tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        target_candidates.append(n.name)
                        break
            except SyntaxError:
                pass

        test_names = _test_function_names(tree)
        if not test_names:
            return GateResult(
                gate_name="relevance",
                passed=False,
                findings=[Finding(
                    severity=Severity.ERROR,
                    code="no_test_functions",
                    message=(
                        "No test functions or test classes found in the "
                        "generated test code (no test_* functions or Test* "
                        "classes)"
                    ),
                    suggestion=(
                        "Ensure the test file defines at least one function "
                        "named test_* or a class named Test* with test_* "
                        "methods"
                    ),
                )],
                details="zero test functions",
            )

        test_names = _test_function_names(tree)
        if not test_names:
            return GateResult(
                gate_name="relevance",
                passed=False,
                findings=[Finding(
                    severity=Severity.ERROR,
                    code="no_test_functions",
                    message=(
                        "No test functions or test classes found in the "
                        "generated test code (no test_* functions or Test* "
                        "classes)"
                    ),
                    suggestion=(
                        "Ensure the test file defines at least one function "
                        "named test_* or a class named Test* with test_* "
                        "methods"
                    ),
                )],
                details="zero test functions",
            )

        signals: dict[str, bool] = {
            "imports_source_module": _imports_source_module(tree, self.source_module),
            "test_name_references_target": False,
            "imports_target_name": False,
        }
        findings: list[Finding] = []

        if target_candidates:
            keywords = []
            for tgt in target_candidates:
                keywords.extend(_candidate_keywords(tgt))
            keywords = list(dict.fromkeys(keywords))  # dedup, preserve order

            for name in test_names:
                nl = name.lower()
                if any(kw in nl for kw in keywords):
                    signals["test_name_references_target"] = True
                    break

            imported = _imported_target_names(tree, self.source_module)
            for tgt in target_candidates:
                if tgt in imported or tgt.lower() in {n.lower() for n in imported}:
                    signals["imports_target_name"] = True
                    break

            # Strong gaming signal: test file *redefines* the target locally
            for tgt in target_candidates:
                if _redefined_target_names(tree, tgt) and not signals["imports_target_name"]:
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        code="target_shadowed_in_tests",
                        message=(
                            f"Target '{tgt}' appears to be redefined in the test "
                            "file rather than imported from source. Tests may be "
                            "validating a fictional implementation."
                        ),
                    ))

        test_names = _test_function_names(tree)
        if not test_names:
            # Zero test_* functions or Test* classes found. This is the
            # "collected 0 items" case -- the LLM produced a file that
            # pytest can't even collect. Flag it here so the repair loop
            # gets a clear message instead of falling through to the
            # sandbox with an empty finding.
            return GateResult(
                gate_name="relevance",
                passed=False,
                findings=[Finding(
                    severity=Severity.ERROR,
                    code="no_test_functions",
                    message=(
                        "No test functions or test classes found in the "
                        "generated test code (no test_* functions or Test* "
                        "classes)"
                    ),
                    suggestion=(
                        "Ensure the test file defines at least one function "
                        "named test_* or a class named Test* with test_* "
                        "methods"
                    ),
                )],
                details="zero test functions",
            )


        passed_signals = sum(1 for v in signals.values() if v)
        passed = passed_signals >= self.min_relevance_signals

        if not passed:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="tests_unrelated_to_source",
                message=(
                    "Generated tests appear unrelated to the function under "
                    f"test. None of {sorted(signals)} succeeded "
                    f"(target={target_candidates or 'unknown'})."
                ),
                suggestion=(
                    f"Ensure tests import from '{self.source_module}' and "
                    "exercise the original function names."
                ),
            ))

        return GateResult(
            gate_name="relevance",
            passed=passed,
            findings=findings,
            details=(
                f"signals={signals}; targets={target_candidates}; "
                f"min_required={self.min_relevance_signals}"
            ),
        )
