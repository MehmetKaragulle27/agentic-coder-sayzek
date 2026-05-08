"""ULT (UnLeakedTestBench) benchmark loader.

Dataset: 3,909 Python functions with high cyclomatic complexity.
Source: https://github.com/huangd1999/UnLeakedTestBench
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/huangd1999/UnLeakedTestBench.git"
LOCAL_DIR_NAME = "UnLeakedTestBench"


class ULTDataset:
    """Loads the ULT benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / LOCAL_DIR_NAME

    @property
    def name(self) -> str:
        return "ult"

    @property
    def language(self) -> Optional[str]:
        return "python"

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    # Preferred dataset files, in priority order. ``ULT.jsonl`` is the
    # canonical split reported in the paper; ``ULT_Lite.jsonl`` is a
    # smaller curated subset; ``PLT.jsonl`` is a related public-leakage
    # set we only fall back to if neither of the others exists.
    _PREFERRED_FILES = ("ULT.jsonl", "ULT_Lite.jsonl", "PLT.jsonl")

    @staticmethod
    def _iter_entries(path: Path):
        """Yield dict entries from ``path`` tolerating both file shapes.

        Despite the ``.jsonl`` extension, UnLeakedTestBench ships the
        datasets as a single pretty-printed JSON array spanning many
        lines. Older snapshots were true JSONL. This helper handles
        both:

          1. Try to parse the whole file as one JSON document; if that
             yields a list, iterate it.
          2. Otherwise fall back to line-by-line parsing, silently
             skipping lines that aren't valid JSON objects.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None

        if isinstance(doc, list):
            for entry in doc:
                if isinstance(entry, dict):
                    yield entry
            return
        if isinstance(doc, dict):
            yield doc
            return

        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in "[]{}":
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        # Pick the best available file by priority, not just the first
        # ``*.jsonl`` we stumble across (``PLT.jsonl`` is intentionally
        # last because it's the leakage-positive control set, not the
        # main benchmark).
        datasets_dir = self._root / "datasets"
        chosen: Optional[Path] = None
        for name in self._PREFERRED_FILES:
            candidate = datasets_dir / name
            if candidate.exists():
                chosen = candidate
                break
        if chosen is None:
            # Legacy layout: anything ending in .jsonl anywhere.
            found = sorted(self._root.rglob("*.jsonl"))
            if found:
                chosen = found[0]

        if chosen is not None:
            log.info("Loading ULT cases from %s", chosen)
            for i, entry in enumerate(self._iter_entries(chosen)):
                code = (
                    entry.get("code")
                    or entry.get("function")
                    or entry.get("canonical_solution")
                    or entry.get("prompt", "")
                )
                if not code or not isinstance(code, str):
                    continue

                # ULT ships a ``prompt`` describing what the function
                # should do; feed that to the coding agent so it
                # generates *semantically meaningful* tests instead of
                # guessing from the signature alone.
                prompt = entry.get("prompt") or entry.get("description") or ""
                user_request = "Generate comprehensive unit tests"
                if prompt:
                    user_request = (
                        "Generate comprehensive unit tests for the function "
                        "below.\n\nProblem description:\n" + prompt.strip()
                    )

                task_id = entry.get("task_id") or str(i)
                func_name = entry.get("func_name") or entry.get("entry_point") or ""
                suffix = f"{task_id}-{func_name}" if func_name else str(task_id)

                cases.append(BenchmarkCase(
                    id=f"ult-{suffix}",
                    code=code,
                    language="python",
                    metadata={
                        k: v for k, v in entry.items()
                        # Skip the long reference tests list so the
                        # per-case JSONs stay readable.
                        if k not in ("tests",)
                    },
                    user_request=user_request,
                ))

        # Last-ditch fallback: plain .py files under the repo (only
        # used if no JSONL datasets are found at all).
        if not cases:
            for py_file in sorted(self._root.rglob("*.py"))[:500]:
                code = py_file.read_text(encoding="utf-8", errors="replace")
                if len(code) < 20:
                    continue
                cases.append(BenchmarkCase(
                    id=f"ult-{py_file.stem}",
                    code=code,
                    language="python",
                    user_request="Generate comprehensive unit tests",
                ))

        log.info("Loaded %d cases from ULT benchmark", len(cases))
        return cases
