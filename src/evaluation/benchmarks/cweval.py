"""CWEval benchmark loader.

Dataset: 119 security-focused coding tasks (25 Python + 23 JS, plus others).
Source: https://github.com/Co1lin/CWEval
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/Co1lin/CWEval.git"
LOCAL_DIR_NAME = "CWEval"


class CWEvalDataset:
    """Loads the CWEval benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, language_filter: Optional[str] = None):
        self._root = data_dir / LOCAL_DIR_NAME
        self._lang_filter = language_filter

    @property
    def name(self) -> str:
        return "cweval"

    @property
    def language(self) -> Optional[str]:
        return self._lang_filter

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    def _detect_language(self, filename: str) -> str:
        if filename.endswith((".js", ".ts", ".jsx", ".tsx")):
            return "javascript"
        return "python"

    # Map language subdirectory -> (language name, task-file glob).
    # CWEval's canonical layout is:
    #   benchmark/core/py/cwe_NNN_M_task.py   -> source to analyze
    #   benchmark/core/py/cwe_NNN_M_test.py   -> reference test harness (SKIP)
    #   benchmark/core/c/ ...                  -> C sources + Python harness
    #                                             that execs compiled binaries
    #                                             (SKIP — our sandbox can't run them)
    # We only load `*_task.{py,js}` from the py/ and js/ subtrees.
    _LANG_DIRS = {
        "py": "python",
        "js": "javascript",
    }

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        # CWEval stores tasks as JSON under benchmark/ or data/
        for json_file in sorted(self._root.rglob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            items = data if isinstance(data, list) else data.get("tasks", data.get("data", []))
            if not isinstance(items, list):
                continue

            for entry in items:
                code = entry.get("code") or entry.get("prompt") or entry.get("source", "")
                if not code:
                    continue
                lang = entry.get("language", "python").lower()
                if lang not in ("python", "javascript"):
                    continue
                if self._lang_filter and lang != self._lang_filter:
                    continue

                cwe = entry.get("cwe") or entry.get("CWE", "")
                cases.append(BenchmarkCase(
                    id=f"cweval-{entry.get('id', len(cases))}",
                    code=code,
                    language=lang,
                    expected_cwe=str(cwe),
                    metadata=entry,
                    user_request="Generate comprehensive unit tests",
                ))

        # Fallback: walk the canonical repo layout and only keep *_task.* files
        # from Python / JavaScript subtrees. This deliberately excludes every
        # *_test.{py,js}, compiled-language harness (c/, cpp/, go/), and any
        # other auxiliary file so the pipeline always sees real source code.
        if not cases:
            core = self._root / "benchmark" / "core"
            if core.exists():
                for sub, lang in self._LANG_DIRS.items():
                    if self._lang_filter and lang != self._lang_filter:
                        continue
                    ext = ".py" if sub == "py" else ".js"
                    base = core / sub
                    if not base.exists():
                        continue
                    for src in sorted(base.rglob(f"*_task{ext}")):
                        code = src.read_text(encoding="utf-8", errors="replace")
                        stem = src.stem  # e.g. "cwe_022_0_task"
                        cwe_id = stem.split("_")[1] if stem.startswith("cwe_") else ""
                        cases.append(BenchmarkCase(
                            id=f"cweval-{stem}",
                            code=code,
                            language=lang,
                            expected_cwe=f"CWE-{cwe_id}" if cwe_id else "",
                            metadata={"file": str(src.relative_to(self._root))},
                            user_request="Generate comprehensive unit tests",
                        ))

        log.info("Loaded %d cases from CWEval benchmark", len(cases))
        return cases
