"""CodeJudgeBench loader.

Dataset: LLM judge evaluation pairs (code, tests, human judgment).
Source: HuggingFace dataset ``mattymchen/codejudgebench`` (Apache 2.0).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)

HF_DATASET_ID = "mattymchen/codejudgebench"
LOCAL_DIR_NAME = "codejudgebench"


class CodeJudgeBenchDataset:
    """Loads the CodeJudgeBench testgen subset."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / LOCAL_DIR_NAME

    @property
    def name(self) -> str:
        return "codejudgebench"

    @property
    def language(self) -> Optional[str]:
        return None  # mixed

    def download(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        marker = self._root / ".downloaded"
        if marker.exists():
            return self._root

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                local_dir=str(self._root),
            )
        except ImportError:
            log.warning(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            return self._root
        except Exception as exc:
            log.warning("Failed to download CodeJudgeBench: %s", exc)
            return self._root

        marker.write_text("ok")
        return self._root

    # CodeJudgeBench ships as parquet files under <subset>/<model>-*.parquet.
    # We use the ``codegen`` subset because its ``pos_response`` column is
    # human-verified-correct code; measuring our pipeline's pass rate on that
    # is a direct false-positive measurement (ideally close to 100%).
    # Each problem (question_id) appears once per model file; we dedupe by
    # question_id so N cases = N unique problems.
    _PREFERRED_SUBSET = "codegen"

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Pull the first fenced ```python ... ``` block from a response.

        CodeJudgeBench ``pos_response``/``neg_response`` values are raw model
        outputs that include prose + fenced code. We want just the code.
        Falls back to the full text if no fence is found.
        """
        if not isinstance(text, str):
            return ""
        import re
        m = re.search(r"```(?:python|py|cpp|c\+\+|c|java|javascript|js)?\s*\n(.*?)```",
                      text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _strip_stdin_driver(code: str) -> str:
        """Remove competitive-programming stdin drivers from module scope.

        CodeJudgeBench solutions wrap logic in ``def solve()`` / ``def main()``
        and then append a top-level driver like::

            t = int(input())
            for _ in range(t):
                n, k = map(int, input().split())
                print(solve(n, k))

        That driver fires at ``import`` time, breaking pytest collection with
        ``OSError: reading from stdin while output is captured``. We keep
        imports, function defs, class defs, and module-level constant
        assignments; everything else at module scope is dropped.
        """
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        keep_types = (ast.Import, ast.ImportFrom, ast.FunctionDef,
                      ast.AsyncFunctionDef, ast.ClassDef)
        driver_builtins = {"input", "print", "sys"}
        new_body: list = []
        dropped_any = False

        for node in tree.body:
            if isinstance(node, keep_types):
                new_body.append(node)
                continue
            # Keep constant assignments (MODULE_CONST = "...", N = 10, etc.)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value if isinstance(node, ast.Assign) else node.value
                if isinstance(value, ast.Constant):
                    new_body.append(node)
                    continue
                # Drop if it calls input/print or references stdin
                src = ast.unparse(value) if value is not None else ""
                if any(b in src for b in driver_builtins):
                    dropped_any = True
                    continue
                new_body.append(node)
                continue
            # Drop top-level loops, expressions, if __name__ blocks with I/O
            node_src = ast.unparse(node)
            if any(b in node_src for b in driver_builtins):
                dropped_any = True
                continue
            new_body.append(node)

        if not dropped_any:
            return code

        tree.body = new_body
        return ast.unparse(tree)

    @staticmethod
    def _ensure_typing_imports(code: str) -> str:
        """Auto-inject ``from typing import ...`` when LeetCode-style code
        uses type annotations like ``List[int]`` without importing them.

        LeetCode's online judge pre-injects ``typing`` symbols; when we copy
        a ``pos_response`` out it often looks like ``class Solution:
        def foo(self, nums: List[int]) -> List[int]`` with no import,
        which breaks under a plain Python interpreter with
        ``NameError: name 'List' is not defined``.
        """
        if "from typing import" in code or "import typing" in code:
            return code
        typing_symbols = ("List", "Dict", "Tuple", "Set", "Optional",
                          "Union", "Iterable", "Callable", "Any", "Deque",
                          "DefaultDict", "FrozenSet", "Sequence", "Mapping")
        import re
        used = [s for s in typing_symbols
                if re.search(rf"\b{s}\[", code)]
        if not used:
            return code
        return f"from typing import {', '.join(used)}\n\n" + code

    @staticmethod
    def _functions_read_stdin(code: str) -> bool:
        """Return True if any function body calls input() or reads sys.stdin.

        Used to skip cases that would hang under pytest when invoked.
        """
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    # Direct input() call
                    if isinstance(func, ast.Name) and func.id == "input":
                        return True
                    # sys.stdin.readline() / sys.stdin.read()
                    if (isinstance(func, ast.Attribute)
                            and isinstance(func.value, ast.Attribute)
                            and isinstance(func.value.value, ast.Name)
                            and func.value.value.id == "sys"
                            and func.value.attr == "stdin"):
                        return True
        return False

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []
        seen_ids: set = set()

        # Prefer the codegen subset (pos_response = human-correct code)
        subset_dir = self._root / self._PREFERRED_SUBSET
        parquet_files = sorted(subset_dir.glob("*.parquet")) if subset_dir.exists() else []

        if parquet_files:
            try:
                import pandas as pd
            except ImportError:
                log.warning(
                    "pandas not installed; cannot parse CodeJudgeBench parquet files. "
                    "Install with: pip install pandas pyarrow"
                )
                return []

            for pq in parquet_files:
                try:
                    df = pd.read_parquet(pq)
                except Exception as exc:
                    log.warning("Failed to read %s: %s", pq.name, exc)
                    continue

                for _, row in df.iterrows():
                    qid = str(row.get("question_id", "")).strip()
                    if not qid or qid in seen_ids:
                        continue
                    raw_response = row.get("pos_response", "")
                    code = self._extract_code_block(raw_response)
                    starter = (row.get("starter_code") or "").strip()
                    if starter and starter not in code:
                        code = starter + "\n\n" + code
                    code = self._strip_stdin_driver(code)
                    code = self._ensure_typing_imports(code)
                    if not code.strip():
                        continue
                    # Skip if no testable function/class survived the strip
                    if "def " not in code and "class " not in code:
                        continue
                    # Skip solutions whose functions read from stdin internally.
                    # These would hang under pytest (e.g. ``def solve(): n = int(input())``).
                    # We only want solutions where logic is parameterised through args.
                    if self._functions_read_stdin(code):
                        continue
                    seen_ids.add(qid)
                    # Pass the problem description into user_request so the
                    # coder agent writes semantically-correct tests instead
                    # of guessing from the function name. Without this the
                    # LLM invents assertions like ``assert foo('WWB', 1) ==
                    # 'YES'`` when the correct answer is 'NO' and reports
                    # spurious failures on human-verified code.
                    title = (row.get("question_title") or "").strip()
                    problem = (row.get("question_content") or "").strip()
                    if problem:
                        request = (
                            "Generate unit tests for the provided reference solution.\n\n"
                            "CRITICAL RULES:\n"
                            "1. Only test inputs that satisfy the problem's stated "
                            "constraints. Do NOT invent out-of-spec inputs (zero/negative "
                            "sizes, empty strings when the spec says length>=1, etc).\n"
                            "2. Derive expected outputs from the problem specification, "
                            "not from the function name. When in doubt, trace the "
                            "reference solution by hand on small inputs.\n"
                            "3. Do NOT wrap assertions in `pytest.raises` unless the "
                            "problem explicitly says invalid input raises an exception.\n"
                            "4. Use small inputs (<= 20 elements) so tests finish quickly.\n\n"
                        )
                        if title:
                            request += f"# Problem: {title}\n\n"
                        request += problem
                    else:
                        request = "Generate comprehensive unit tests"

                    cases.append(BenchmarkCase(
                        id=f"cjb-{qid}",
                        code=code,
                        language="python",  # CJB solutions are Python unless starter says otherwise
                        metadata={
                            "source_model": pq.stem.replace("-00000-of-00001", ""),
                            "platform": row.get("platform"),
                            "difficulty": row.get("difficulty"),
                            "question_title": row.get("question_title"),
                        },
                        user_request=request,
                    ))

        # Legacy fallback: JSONL (older CJB releases)
        if not cases:
            for jsonl in sorted(self._root.rglob("*.jsonl"))[:3]:
                for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = entry.get("code") or entry.get("prompt", "")
                    if not code:
                        continue
                    cases.append(BenchmarkCase(
                        id=f"cjb-{i}",
                        code=code,
                        language=entry.get("language", "python").lower(),
                        metadata=entry,
                        user_request="Generate comprehensive unit tests",
                    ))

        log.info("Loaded %d cases from CodeJudgeBench (%s subset)",
                 len(cases), self._PREFERRED_SUBSET)
        return cases
