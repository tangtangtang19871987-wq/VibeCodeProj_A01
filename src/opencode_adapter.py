"""The one place that knows how to invoke OpenCode.

Every other module — every example graph — talks to OpenCode only through
`OpenCodeAdapter.run(task)`. Nothing outside this file constructs a
subprocess command line, an HTTP request, or parses OpenCode's output.

Three backends:

- `SDKBackend` talks to a running `opencode serve` server through the
  official Python client, `opencode-ai` on PyPI
  (github.com/sst/opencode-sdk-python — see research/sources.md). This is
  the preferred backend where available: structured, typed responses
  (session objects, message/part lists) instead of scraping CLI stdout.
- `CLIBackend` shells out to the `opencode` CLI (`opencode run`, per
  packages/opencode/src/cli/cmd/run.ts — see research/sources.md) in
  non-interactive, single-shot mode. Useful when only the CLI binary is
  available and no server is running.
- `FakeBackend` is an explicit, honest stand-in used only so the examples
  in this repository run end-to-end without requiring OpenCode (a server,
  the CLI binary, or a model API key) to be installed at all. It is NOT a
  reimplementation of OpenCode's coding-agent loop (see docs/architecture.md
  and research/project_comparison.md for why that would defeat the point of
  this repository) — it cannot fix anything on its own. It just runs a
  caller-supplied "simulate one attempt" function in a bounded loop and
  checks a caller-supplied command between attempts, which is enough to
  demonstrate the LangGraph-facing contract (turns, retries, a growing
  session log) without pretending to be a general coding agent.

Swapping between backends requires no change to any LangGraph code —
that's the point of the adapter existing at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from src.contracts import AgentResult, AgentTask

_LOG_FILENAME = "opencode_session.log"
_DEFAULT_SDK_BASE_URL = "http://localhost:54321"


class SDKBackend:
    """Talks to a running `opencode serve` server via the official
    `opencode-ai` Python client.

    This backend only holds a conversation with an already-running server
    (default `http://localhost:54321`, or `OPENCODE_BASE_URL`) — it does
    not start, stop, or otherwise manage that server's process, exactly as
    `CLIBackend` doesn't install the `opencode` binary. Run `opencode serve`
    yourself (or point `OPENCODE_BASE_URL` at one already running) before
    selecting this backend.

    Session isolation (docs/context_management.md): `execute()` always
    calls `client.session.create()` itself and never accepts an externally
    supplied session id, so every `AgentTask` gets a brand-new, empty
    OpenCode session — the same guarantee `CLIBackend` gets by never
    passing `--continue`/`--session`/`--fork`, just made explicit by the
    session object this backend gets back.

    IMPORTANT — workspace scoping is NOT free here, unlike `CLIBackend`.
    `CLIBackend` runs `opencode` with `cwd=task.workspace`, so the
    workspace boundary (docs/sandboxing.md, docs/bounded_autonomy.md) comes
    for free from the subprocess call itself. The installed `opencode-ai`
    client (v0.1.0-alpha.36 at the time this was written — verified by
    reading the installed package's source, since no live server was
    reachable in this environment to test against) has no per-session or
    per-request directory parameter at all: a server's working directory
    (`client.app.get().path.cwd`) is fixed for the lifetime of that
    `opencode serve` process. `execute()` checks this on every call and
    fails closed — returns an unambiguous error instead of silently running
    against the wrong directory — if the server's cwd doesn't match
    `task.workspace`. In practice this means: **one `opencode serve`
    process per distinct workspace** (launch it with
    `cwd=task.workspace` yourself, and point `OPENCODE_BASE_URL` /
    `base_url` at that instance), not one long-lived shared server handling
    every task. If a future SDK version adds real per-session directory
    scoping, this check can be relaxed — don't remove it speculatively
    before then.
    """

    def __init__(self, base_url: str | None = None) -> None:
        try:
            from opencode_ai import Opencode  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "SDKBackend requires the 'opencode-ai' package: pip install --pre opencode-ai"
            ) from exc
        base_url = base_url or os.environ.get("OPENCODE_BASE_URL", _DEFAULT_SDK_BASE_URL)
        self._client = Opencode(base_url=base_url)

    def execute(self, task: AgentTask) -> AgentResult:
        task.workspace.mkdir(parents=True, exist_ok=True)
        log_path = task.workspace / _LOG_FILENAME

        provider_id = task.context.get("provider_id") or os.environ.get("OPENCODE_PROVIDER")
        model_id = task.context.get("model_id") or os.environ.get("OPENCODE_MODEL")
        if not provider_id or not model_id:
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="SDKBackend needs a provider_id and model_id",
                turns_used=0,
                log_path=None,
                error=(
                    "set AgentTask.context['provider_id']/['model_id'] or the "
                    "OPENCODE_PROVIDER/OPENCODE_MODEL env vars — this backend "
                    "never guesses a model, see class docstring"
                ),
            )

        try:
            server_cwd = Path(self._client.app.get().path.cwd).resolve()
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="could not reach the opencode server",
                turns_used=0,
                log_path=None,
                error=str(exc),
            )
        if server_cwd != task.workspace.resolve():
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="opencode server is not scoped to this task's workspace",
                turns_used=0,
                log_path=None,
                error=(
                    f"server cwd is {server_cwd}, but AgentTask.workspace is "
                    f"{task.workspace.resolve()} — this SDK version has no per-session "
                    "directory scoping (see SDKBackend's docstring); run a dedicated "
                    "`opencode serve` with cwd=task.workspace instead of sharing one server"
                ),
            )

        session = self._client.session.create()  # a fresh session every call — see class docstring
        _append_log(log_path, f"[opencode-sdk] session {session.id} instruction={task.instruction!r}")

        try:
            self._client.session.chat(
                session.id,
                provider_id=provider_id,
                model_id=model_id,
                parts=[{"type": "text", "text": task.instruction}],
                timeout=task.timeout_seconds,
            )
        except Exception as exc:
            _append_log(log_path, f"[opencode-sdk] session {session.id} errored: {exc}")
            return AgentResult(
                task_id=task.task_id,
                claimed_done=False,
                summary="opencode SDK call failed",
                turns_used=0,
                log_path=log_path,
                error=str(exc),
            )

        messages = self._client.session.messages(session.id)
        # The full message/part transcript goes to disk, never into
        # AgentResult — same context-quarantine rule as the other backends
        # (docs/context_management.md).
        _append_log(log_path, f"[opencode-sdk] transcript:\n{messages}")

        reply_text = _extract_reply_text(messages)
        last_assistant_error = _last_assistant_error(messages)
        return AgentResult(
            task_id=task.task_id,
            claimed_done=last_assistant_error is None,
            summary=(reply_text or "(no text reply)")[:280],
            turns_used=sum(1 for item in messages if getattr(item.info, "role", None) == "assistant"),
            log_path=log_path,
            error=str(last_assistant_error) if last_assistant_error else None,
        )


def _extract_reply_text(messages) -> str:
    """Concatenates the text parts of the last assistant message."""
    for item in reversed(messages):
        if getattr(item.info, "role", None) == "assistant":
            texts = [part.text for part in item.parts if getattr(part, "type", None) == "text"]
            if texts:
                return "\n".join(texts)
    return ""


def _last_assistant_error(messages):
    for item in reversed(messages):
        if getattr(item.info, "role", None) == "assistant":
            return getattr(item.info, "error", None)
    return None


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


def _sdk_server_reachable(base_url: str, timeout: float = 0.3) -> bool:
    try:
        import httpx  # noqa: PLC0415 — already a dependency of opencode-ai
    except ImportError:
        return False
    try:
        httpx.get(f"{base_url}/session", timeout=timeout)
        return True
    except httpx.HTTPError:
        return False


class OpenCodeAdapter:
    """Chooses a backend and runs an AgentTask through it.

    backend: "auto" (default) prefers SDKBackend if `opencode-ai` is
    installed AND a server is actually reachable at OPENCODE_BASE_URL
    (default http://localhost:54321); otherwise CLIBackend if the
    `opencode` binary is on PATH; otherwise FakeBackend. Pass "sdk", "cli",
    or "fake" to force one. The OPENCODE_ADAPTER_BACKEND environment
    variable overrides the constructor default, so an example can be run
    against the real thing without editing any code:
    `OPENCODE_ADAPTER_BACKEND=sdk python ...` (with `opencode serve`
    already running) or `OPENCODE_ADAPTER_BACKEND=cli python ...`.
    """

    def __init__(self, backend: str = "auto") -> None:
        backend = os.environ.get("OPENCODE_ADAPTER_BACKEND", backend)
        if backend == "auto":
            base_url = os.environ.get("OPENCODE_BASE_URL", _DEFAULT_SDK_BASE_URL)
            if _sdk_server_reachable(base_url):
                backend = "sdk"
            elif shutil.which("opencode"):
                backend = "cli"
            else:
                backend = "fake"
        if backend not in ("sdk", "cli", "fake"):
            raise ValueError(f"unknown backend: {backend!r}")
        self._backend = {"sdk": SDKBackend, "cli": CLIBackend, "fake": FakeBackend}[backend]()
        self.backend_name = backend

    def run(self, task: AgentTask) -> AgentResult:
        return self._backend.execute(task)
