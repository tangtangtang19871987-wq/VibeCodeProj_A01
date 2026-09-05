"""The one place that knows how to invoke OpenCode.

Every other module — every example graph — talks to OpenCode only through
`OpenCodeAdapter.run(task)`. Nothing outside this file constructs a
subprocess command line or parses OpenCode's output.

Two backends:

- `CLIBackend` shells out to the real `opencode` CLI (`opencode run`, per
  packages/opencode/src/cli/cmd/run.ts — see research/sources.md) in
  non-interactive, single-shot mode. This is what a real deployment uses.
- `FakeBackend` is an explicit, honest stand-in used only so the examples
  in this repository run end-to-end without requiring the real `opencode`
  binary and a model API key to be installed. It is NOT a reimplementation
  of OpenCode's coding-agent loop (see docs/architecture.md and
  research/project_comparison.md for why that would defeat the point of
  this repository) — it cannot fix anything on its own. It just runs a
  caller-supplied "simulate one attempt" function in a bounded loop and
  checks a caller-supplied command between attempts, which is enough to
  demonstrate the LangGraph-facing contract (turns, retries, a growing
  session log) without pretending to be a general coding agent.

Swapping `FakeBackend` for `CLIBackend` requires no change to any
LangGraph code — that's the point of the adapter existing at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from src.contracts import AgentResult, AgentTask

_LOG_FILENAME = "opencode_session.log"


class CLIBackend:
    """Invokes the real `opencode` CLI in non-interactive mode.

    Deliberately never passes `--continue`, `--session`, or `--fork`
    (packages/opencode/src/cli/cmd/run.ts — see research/sources.md).
    Omitting them is what makes every call start a brand-new OpenCode
    session with empty history: no prior task's conversation, file reads,
    or shell output is visible to this one. `AgentTask` (src/contracts.py)
    has no session-id field at all, so there is nothing a caller could pass
    in even by accident to opt back into continuation — see
    docs/context_management.md's "session isolation" section for the full
    argument.
    """

    def execute(self, task: AgentTask) -> AgentResult:
        task.workspace.mkdir(parents=True, exist_ok=True)
        log_path = task.workspace / _LOG_FILENAME
        fixed_args = ["opencode", "run", "--format", "json"]  # never add -c/--continue/--session/--fork here
        cmd = [*fixed_args, task.instruction]
        assert not any(a.startswith(("--continue", "--session", "--fork", "-c", "-s")) for a in fixed_args), (
            "CLIBackend must never continue a prior OpenCode session — see class docstring"
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=task.workspace,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _append_log(log_path, f"[opencode] timed out after {task.timeout_seconds}s")
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="OpenCode timed out",
                turns_used=0,
                log_path=log_path,
                error=str(exc),
            )
        except FileNotFoundError as exc:
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="opencode binary not found on PATH",
                turns_used=0,
                log_path=None,
                error=str(exc),
            )

        _append_log(log_path, proc.stdout)
        if proc.stderr:
            _append_log(log_path, proc.stderr)

        # Non-interactive `opencode run --format json` emits one JSON event
        # per line; a rough turn count is enough here since the examples
        # only need "did it run, roughly how much work did it do."
        turns_used = sum(1 for line in proc.stdout.splitlines() if line.strip())
        claimed_done = proc.returncode == 0
        summary = (
            proc.stdout.strip().splitlines()[-1][:280]
            if claimed_done and proc.stdout.strip()
            else f"opencode exited with code {proc.returncode}"
        )
        return AgentResult(
            task_id=task.task_id,
            claimed_done=claimed_done,
            summary=summary,
            turns_used=turns_used,
            log_path=log_path,
            error=None if claimed_done else proc.stderr[-500:],
        )


class FakeBackend:
    """A teaching stand-in for `CLIBackend`. See module docstring."""

    def execute(self, task: AgentTask) -> AgentResult:
        task.workspace.mkdir(parents=True, exist_ok=True)
        log_path = task.workspace / _LOG_FILENAME

        simulate_attempt = task.context.get("simulate_attempt")
        check_command = task.context.get("check_command")

        _append_log(
            log_path,
            f"[fake-opencode] task={task.task_id} instruction={task.instruction!r} "
            f"permissions={task.permissions} max_turns={task.max_turns}",
        )

        last_error = ""
        claimed_done = False
        attempt = 0
        while attempt < task.max_turns:
            attempt += 1
            if simulate_attempt is not None:
                note = simulate_attempt(task.workspace, attempt, last_error)
                _append_log(log_path, f"[fake-opencode] attempt {attempt}: {note}")
            else:
                _append_log(log_path, f"[fake-opencode] attempt {attempt}: no-op (no simulate_attempt given)")

            if check_command is None:
                claimed_done = True
                break

            # See src/verifier.py's _NO_BYTECODE_CACHE_ENV comment: fast
            # rewrite-then-check cycles can otherwise reuse a stale .pyc.
            result = subprocess.run(
                check_command,
                cwd=task.workspace,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = (result.stdout + result.stderr).strip()
            _append_log(
                log_path,
                f"[fake-opencode] check_command exit={result.returncode}\n{output}",
            )
            if result.returncode == 0:
                claimed_done = True
                break
            last_error = output[-2000:]

        summary = (
            f"finished after {attempt} attempt(s)"
            if claimed_done
            else f"gave up after {attempt} attempt(s): {last_error[-200:]}"
        )
        return AgentResult(
            task_id=task.task_id,
            claimed_done=claimed_done,
            summary=summary,
            turns_used=attempt,
            log_path=log_path,
            error=None if claimed_done else "verifier check_command still failing",
        )


def _append_log(log_path: Path, text: str) -> None:
    if not text:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"--- {time.strftime('%H:%M:%S')} ---\n{text}\n")


class OpenCodeAdapter:
    """Chooses a backend and runs an AgentTask through it.

    backend: "auto" (default) uses CLIBackend if the `opencode` binary is
    on PATH, otherwise falls back to FakeBackend. Pass "cli" or "fake" to
    force one. The OPENCODE_ADAPTER_BACKEND environment variable overrides
    the constructor default, so an example can be run against the real CLI
    without editing any code: `OPENCODE_ADAPTER_BACKEND=cli python ...`.
    """

    def __init__(self, backend: str = "auto") -> None:
        backend = os.environ.get("OPENCODE_ADAPTER_BACKEND", backend)
        if backend == "auto":
            backend = "cli" if shutil.which("opencode") else "fake"
        if backend not in ("cli", "fake"):
            raise ValueError(f"unknown backend: {backend!r}")
        self._backend = CLIBackend() if backend == "cli" else FakeBackend()
        self.backend_name = backend

    def run(self, task: AgentTask) -> AgentResult:
        return self._backend.execute(task)
