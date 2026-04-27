# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution and feedback helpers for ColBench (Backend) live-loop runs.

ColBench (Meta FAIR, SWEET-RL, March 2025; HF: ``facebook/collaborative_agent_bench``)
is a multi-turn coding benchmark where the agent solves a Python task by asking
clarifying questions to a *simulated human* that has access to the reference
solution and the hidden tests. The agent submits final code as a fenced
``\`\`\`python`` block (or with the explicit ``<code>...</code>`` marker the
SWEET-RL release uses); the loop terminates on first submission.

This module mirrors ``convcodeworld/executor.py``: it lifts the generic
sandboxed-execution primitives (compile, sandboxed subprocess with rlimit,
optional ``unshare -n``, byte-level-tokenizer artefact normalisation,
longest-compilable-prefix recovery) and replaces only the benchmark-specific
layers - the test-harness builder (ColBench tasks declare ``private_tests`` as
Python ``unittest`` source rather than BigCodeBench-style ``check`` callables),
the simulator-prompt builder (the simulator gets the reference solution +
hidden tests, the agent's latest question, and the prior dialogue), and the
submission detector.
"""

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
_SUBPROCESS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

# Markers the agent uses to signal "I am submitting code now". The fenced
# ``\`\`\`python`` block is the canonical SWEET-RL convention; the explicit
# ``<code>...</code>`` and ``<submit>...</submit>`` tags are alternates seen in
# the upstream prompts. The detector treats *any* fenced block as a submission
# rather than a clarifying question.
_SUBMIT_TAG_PATTERNS = (
    re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<submit>(.*?)</submit>", re.DOTALL | re.IGNORECASE),
)
_FENCED_PYTHON_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


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
    for pattern in _SUBMIT_TAG_PATTERNS:
        match = pattern.search(cleaned)
        if match is not None:
            return match.group(1).strip()
    match = _FENCED_PYTHON_RE.search(cleaned)
    if match is None:
        match = re.search(r"```[ \t]*\n(.*?)```", cleaned, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return cleaned.replace("```", "").strip()


def detect_submission(text: str) -> str | None:
    """Return the submitted code body if ``text`` is a submission, else None.

    A submission is signalled by either a ``<code>`` / ``<submit>`` tag or a
    fenced ``\`\`\`python`` block. Plain natural-language replies (clarifying
    questions, confusion, etc.) return ``None`` so the live loop can keep
    asking.
    """
    if not text:
        return None
    normalized = normalize_tokenizer_artifacts(text)
    for pattern in _SUBMIT_TAG_PATTERNS:
        match = pattern.search(normalized)
        if match is not None:
            body = match.group(1).strip()
            if body:
                return body
    match = _FENCED_PYTHON_RE.search(normalized)
    if match is not None:
        body = match.group(1).strip()
        if body:
            return body
    return None


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
    """Normalize a model-generated candidate against the task schema.

    ColBench tasks may carry ``code_prompt`` (a stub with imports + signature),
    ``starter_code``, or just an ``entry_point`` string. If the model returned
    only a function body and we have the stub, prepend it; then return the
    longest compilable prefix so a truncated generation still earns partial
    credit when its head defines the entry point.
    """
    code = extract_code(code)
    code_prompt = str(
        task_get(task, "code_prompt", "")
        or task_get(task, "starter_code", "")
        or ""
    )
    entry_point = str(task_get(task, "entry_point", "") or "")
    if code_prompt and entry_point and f"def {entry_point}" not in code:
        separator = "" if code_prompt.endswith((" ", "\t", "\n")) else "\n"
        code = code_prompt + separator + code
    return _longest_compilable_prefix(code, entry_point)


def _normalize_private_tests(task: Mapping[str, Any] | Any) -> str:
    """Return the private-test source as a single Python string.

    ColBench's HF schema is not 100% stable across snapshots, so accept all of:
      - ``private_tests`` as a string of Python ``unittest`` source
      - ``private_tests`` as a list of Python source strings (concatenated)
      - ``test`` (BigCodeBench-style) as a string with a ``check(func)`` callable
      - ``hidden_tests`` as either form
    """
    for key in ("private_tests", "hidden_tests", "test"):
        value = task_get(task, key, None)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "\n\n".join(str(item) for item in value if item)
    return ""


def build_test_script(task: Mapping[str, Any] | Any, candidate_code: str) -> str:
    test = _normalize_private_tests(task)
    entry_point = str(task_get(task, "entry_point", "") or "")
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
    ) if entry_point else textwrap.dedent(
        """
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


def _subprocess_env() -> dict[str, str]:
    path_parts = [str(Path(sys.executable).parent)]
    existing_path = os.environ.get("PATH", "")
    if existing_path:
        path_parts.append(existing_path)
    return {
        **_SUBPROCESS_THREAD_ENV,
        "PATH": os.pathsep.join(path_parts),
        "PYTHONNOUSERSITE": "1",
        "MPLBACKEND": "Agg",
    }


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
    """Compile and execute a candidate against the task's hidden tests."""
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
    env = _subprocess_env()
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


def _task_description(task: Mapping[str, Any] | Any) -> str:
    """Return the natural-language description shown to the agent."""
    return str(
        task_get(task, "description")
        or task_get(task, "instruction")
        or task_get(task, "instruct_prompt")
        or task_get(task, "prompt")
        or task_get(task, "complete_prompt")
        or ""
    )


def _reference_solution(task: Mapping[str, Any] | Any) -> str:
    return str(
        task_get(task, "reference_solution")
        or task_get(task, "canonical_solution")
        or task_get(task, "solution")
        or task_get(task, "code")
        or ""
    )


def build_simulator_prompt(
    task: Mapping[str, Any] | Any,
    agent_question: str,
    dialogue_so_far: str,
    *,
    max_chars: int = 6000,
) -> str:
    """Build the prompt for the human-simulator (Gemma4) feedback model.

    The simulator has access to the reference solution and hidden tests. It
    answers the agent's *latest* clarifying question in 1-3 sentences of
    natural language. It must NOT reveal the reference code, write code, or
    quote the hidden tests.
    """
    description = _task_description(task)
    reference = _reference_solution(task)
    private_tests = _normalize_private_tests(task)
    return (
        "You are a ColBench human collaborator simulating a non-coder user.\n"
        "You have access to the private reference solution and hidden tests below.\n"
        "Answer the agent's latest clarifying question in 1-3 plain-English sentences.\n"
        "Never quote, paraphrase, or hint at the reference code or hidden tests.\n"
        "Never write Python code. Never reveal that hidden tests exist.\n"
        "If the agent's question cannot be answered without revealing the solution, "
        "redirect: ask them to try an attempt and you will react.\n"
        "\n### Task description (the user's goal)\n"
        f"{trim_feedback(description, max_chars=max_chars)}\n"
        "\n### Private reference solution (do not reveal)\n"
        f"```python\n{trim_feedback(reference, max_chars=max_chars)}\n```\n"
        "\n### Private hidden tests (do not reveal)\n"
        f"```python\n{trim_feedback(private_tests, max_chars=max_chars)}\n```\n"
        "\n### Dialogue so far (most recent last)\n"
        f"{trim_feedback(dialogue_so_far, max_chars=max_chars) or '(no prior turns)'}\n"
        "\n### Agent's latest question\n"
        f"{trim_feedback(agent_question, max_chars=max_chars)}\n"
        "\n### Your reply (1-3 sentences, no code, no hidden-test details)\n"
    )


def build_feedback_after_submit(
    result: ExecutionResult,
    *,
    max_chars: int = 6000,
) -> str:
    """Format the post-submission failure feedback shown back to the agent.

    Used when ``early_stop_on_pass`` is False (or omitted) and we let the
    agent see the test outcome to refine on the next turn. Mirrors
    ``convcodeworld/executor.py:build_feedback`` minus the verbal block.
    """
    sections: list[str] = []
    sections.append("Compilation feedback:\n" + trim_feedback(result.compilation_feedback, max_chars=max_chars))
    sections.append("Execution feedback:\n" + trim_feedback(result.execution_feedback, max_chars=max_chars))
    if result.passed:
        verbal = "The submitted code passed the hidden tests."
    elif result.status == "compile_error":
        verbal = "The submission did not compile. Fix the syntax/import/name issue and submit again."
    elif result.status == "timeout":
        verbal = "The submission timed out. Avoid non-terminating loops or excessive work."
    else:
        verbal = "The submission failed the hidden tests. Use the failing-test output to refine and submit again."
    sections.append("Outcome:\n" + verbal)
    return "\n\n".join(sections).strip()
