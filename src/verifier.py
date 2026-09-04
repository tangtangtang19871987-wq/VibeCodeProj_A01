"""Deterministic, OpenCode-independent success checks.

See docs/verification.md: the autonomous kernel must not define its own
success. Every function here takes a workspace and returns a
VerificationResult — never an AgentResult, never anything OpenCode itself
produced. Kept deliberately boring: a verifier that needs an LLM to
interpret its result has just moved the "who decides success" problem
somewhere less visible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.contracts import VerificationResult

# A verifier's whole job is to catch code that just changed. Python's
# compiled-bytecode cache keys invalidation off source mtime, and on fast
# edit-then-check cycles (exactly what this repository's retry loops do —
# see docs/failure_recovery.md) two writes can land within the filesystem's
# mtime resolution window, letting a stale .pyc quietly get reused. Every
# subprocess a verifier spawns must run with bytecode caching off.
_NO_BYTECODE_CACHE_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def run_command(
    cmd: list[str], cwd: Path, timeout: int = 30
) -> VerificationResult:
    """Runs a command; passes iff it exits zero."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_NO_BYTECODE_CACHE_ENV,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            passed=False,
            reason=f"command timed out after {timeout}s: {' '.join(cmd)}",
            details={"cmd": cmd},
        )
    output = (result.stdout + result.stderr).strip()
    return VerificationResult(
        passed=result.returncode == 0,
        reason=(
            "command succeeded"
            if result.returncode == 0
            else f"command exited {result.returncode}"
        ),
        details={"cmd": cmd, "output": output[-4000:], "returncode": result.returncode},
    )


def run_pytest(
    cwd: Path, args: list[str] | None = None, timeout: int = 60
) -> VerificationResult:
    """Runs pytest in `cwd`; passes iff the whole suite exits zero."""
    cmd = [sys.executable, "-m", "pytest", *(args or [])]
    return run_command(cmd, cwd=cwd, timeout=timeout)


def file_contains(path: Path, substring: str) -> VerificationResult:
    """Passes iff `path` exists and contains `substring`."""
    if not path.exists():
        return VerificationResult(
            passed=False, reason=f"{path} does not exist", details={"path": str(path)}
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    found = substring in text
    return VerificationResult(
        passed=found,
        reason=(
            f"found {substring!r} in {path.name}"
            if found
            else f"{substring!r} not found in {path.name}"
        ),
        details={"path": str(path)},
    )


def all_of(*results: VerificationResult) -> VerificationResult:
    """Combines several checks; passes only if every one did."""
    failed = [r for r in results if not r.passed]
    if not failed:
        return VerificationResult(passed=True, reason="all checks passed")
    return VerificationResult(
        passed=False,
        reason="; ".join(r.reason for r in failed),
        details={"failed_count": len(failed), "total": len(results)},
    )
