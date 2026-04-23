# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution and deterministic feedback helpers for ConvCodeWorld live-loop runs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


NO_SYNTAX_ERRORS = "No syntax errors"
PASSED_ALL_TEST_RUNS = "Passed all test runs"
_BYTE_LEVEL_ARTIFACTS = str.maketrans({"Ċ": "\n", "Ġ": " ", "ĉ": "\t"})


@dataclass
class ExecutionResult:
    task_id: str
    passed: bool
    status: str
    compilation_feedback: str
    execution_feedback: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_get(task: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(key, default)
    return getattr(task, key, default)


def normalize_tokenizer_artifacts(text: str) -> str:
    return str(text or "").translate(_BYTE_LEVEL_ARTIFACTS)


def extract_code(text: str) -> str:
    """Extract Python code from a markdown block, falling back to raw text."""
    if not text:
        return ""
    text = normalize_tokenizer_artifacts(text)
    cleaned = text.split("\n\n---\n\n", 1)[0]
    match = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        match = re.search(r"```[ \t]*\n(.*?)```", cleaned, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return cleaned.replace("```", "").strip()


def _contains_entry_point(code: str, entry_point: str) -> bool:
    return bool(entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\b", code))


def _longest_compilable_prefix(code: str, entry_point: str = "") -> str:
    lines = code.splitlines()
    start_indices = [0]
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "class ")):
            start_indices.append(index)
        elif entry_point and re.match(rf"def\s+{re.escape(entry_point)}\b", stripped):
            start_indices.append(index)

    seen_starts: set[int] = set()
    for start in start_indices:
        if start in seen_starts:
            continue
        seen_starts.add(start)
        for end in range(len(lines), start, -1):
            prefix = "\n".join(lines[start:end]).strip()
            if not prefix:
                continue
            if entry_point and not _contains_entry_point(prefix, entry_point):
                continue
            try:
                compile(prefix + "\n", "candidate.py", mode="exec")
            except Exception:
                continue
            return prefix
    return code


def compile_code(code: str) -> str:
    try:
        compile(code, "candidate.py", mode="exec")
    except Exception:
        return traceback.format_exc()
    return NO_SYNTAX_ERRORS


def normalize_candidate_code(task: Mapping[str, Any] | Any, code: str) -> str:
    """
    BigCodeBench tasks often provide a ``code_prompt`` containing imports and the
    function signature. If the model emits only the function body, prepend that
    prompt; if it emitted a full function, keep it unchanged.
    """
    code = extract_code(code)
    code_prompt = str(task_get(task, "code_prompt", "") or "")
    entry_point = str(task_get(task, "entry_point", "") or "")
    if code_prompt and entry_point and f"def {entry_point}" not in code:
        separator = "" if code_prompt.endswith((" ", "\t", "\n")) else "\n"
        code = code_prompt + separator + code
    return _longest_compilable_prefix(code, entry_point)


def build_test_script(task: Mapping[str, Any] | Any, candidate_code: str) -> str:
    test = str(task_get(task, "test", "") or "")
    entry_point = str(task_get(task, "entry_point", "") or "")
    call_check = ""
    if entry_point:
        call_check = textwrap.dedent(
            f"""
            if "check" in globals():
                check({entry_point})
            elif "test_check" in globals():
                test_check()
            else:
                suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
                if suite.countTestCases():
                    result = unittest.TextTestRunner(verbosity=2).run(suite)
                    if not result.wasSuccessful():
                        raise AssertionError("unit tests failed")
            """
        )
    return "\n\n".join(
        [
            "import faulthandler\nimport sys\nimport unittest\nfaulthandler.enable()",
            candidate_code,
            test,
            call_check,
        ]
    )


def _limit_resources(memory_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
    except Exception:
        return
    if memory_mb > 0:
        memory_bytes = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    if cpu_seconds > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1))


@lru_cache(maxsize=1)
def _can_unshare_network() -> bool:
    if os.name != "posix" or shutil.which("unshare") is None:
        return False
    try:
        probe = subprocess.run(
            ["unshare", "-n", "--", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except Exception:
        return False
    return probe.returncode == 0


def run_candidate(
    task: Mapping[str, Any] | Any,
    candidate_text: str,
    *,
    timeout_s: int = 30,
    memory_mb: int = 1024,
    network_isolation: str = "auto",
    work_dir: str | Path | None = None,
) -> ExecutionResult:
    """Compile and execute a candidate against the task's embedded tests."""
    task_id = str(task_get(task, "task_id", "unknown"))
    candidate_code = normalize_candidate_code(task, candidate_text)
    compilation_feedback = compile_code(candidate_code)
    if compilation_feedback != NO_SYNTAX_ERRORS:
        return ExecutionResult(
            task_id=task_id,
            passed=False,
            status="compile_error",
            compilation_feedback=compilation_feedback,
            execution_feedback="Skipped execution because candidate code did not compile.",
        )

    script = build_test_script(task, candidate_code)
    env = {"PYTHONNOUSERSITE": "1", "MPLBACKEND": "Agg"}
    use_unshare = network_isolation == "unshare" or (network_isolation == "auto" and _can_unshare_network())

    with tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None) as tmp:
        script_path = Path(tmp) / "candidate_test.py"
        script_path.write_text(script, encoding="utf-8")
        cmd = [sys.executable, str(script_path)]
        if use_unshare:
            cmd = ["unshare", "-n", "--"] + cmd
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                preexec_fn=(lambda: _limit_resources(memory_mb, timeout_s)) if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                task_id=task_id,
                passed=False,
                status="timeout",
                compilation_feedback=NO_SYNTAX_ERRORS,
                execution_feedback=f"Timed out after {timeout_s} seconds.",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode == 0:
        return ExecutionResult(
            task_id=task_id,
            passed=True,
            status="pass",
            compilation_feedback=NO_SYNTAX_ERRORS,
            execution_feedback=PASSED_ALL_TEST_RUNS,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        )

    details = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
    return ExecutionResult(
        task_id=task_id,
        passed=False,
        status="runtime_error",
        compilation_feedback=NO_SYNTAX_ERRORS,
        execution_feedback=details or f"Process exited with code {proc.returncode}.",
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode,
    )


def trim_feedback(text: str, *, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...\n" + text[-half:].lstrip()


def build_feedback(
    result: ExecutionResult,
    *,
    include_compilation: bool = True,
    include_execution: bool = True,
    include_verbal: bool = True,
    max_chars: int = 6000,
) -> str:
    """Create the next user/environment turn from the previous execution result."""
    sections: list[str] = []
    if include_compilation:
        sections.append("Compilation feedback:\n" + trim_feedback(result.compilation_feedback, max_chars=max_chars))
    if include_execution:
        sections.append("Execution feedback:\n" + trim_feedback(result.execution_feedback, max_chars=max_chars))
    if include_verbal:
        if result.passed:
            verbal = "The previous code passed the available tests. Keep the solution unchanged."
        elif result.status == "compile_error":
            verbal = "Revise the code to fix the syntax, import, or name error reported above."
        elif result.status == "timeout":
            verbal = "Revise the code to avoid non-termination or excessive work."
        else:
            verbal = "Revise the code using the failing test output above; preserve the required function signature."
        sections.append("Verbal feedback:\n" + verbal)
    return "\n\n".join(sections).strip()
