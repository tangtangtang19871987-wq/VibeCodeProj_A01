"""The contract between LangGraph (control plane) and OpenCode (autonomous kernel).

See docs/architecture.md and docs/bounded_autonomy.md. Everything that
crosses the LangGraph <-> OpenCode boundary is one of the three types below.
Nothing else should cross it — see docs/context_management.md for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# What OpenCode is allowed to do inside its workspace. This is advisory to
# the adapter, which maps it onto real flags/sandboxing for the backend in
# use — see docs/sandboxing.md ("a permission profile is a policy, not a
# promise": something has to actually enforce it).
PermissionProfile = Literal["read_only", "read_write", "read_write_exec"]


@dataclass(frozen=True)
class AgentTask:
    """A bounded unit of work LangGraph hands to the OpenCode kernel.

    This is the entire interface OpenCode sees. It cannot ask for more
    context, more workspace, more turns, or more permission than is set
    here — those are LangGraph's decisions (docs/bounded_autonomy.md).
    """

    task_id: str
    instruction: str
    workspace: Path
    context: dict[str, Any] = field(default_factory=dict)
    permissions: PermissionProfile = "read_write"
    max_turns: int = 8
    timeout_seconds: int = 120
    # Free-form, human-readable statement of intent. Not machine-checked —
    # the actual check is a VerificationResult produced by src/verifier.py.
    # Kept here only so a human reading a task/result pair understands what
    # was being attempted.
    success_criteria: str = ""


@dataclass(frozen=True)
class AgentResult:
    """What OpenCode hands back to LangGraph after an AgentTask.

    `claimed_done` is named that way, not `success`, on purpose: it is
    OpenCode's own claim about itself and must never be treated as ground
    truth. See docs/verification.md.
    """

    task_id: str
    claimed_done: bool
    summary: str
    turns_used: int
    log_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """The output of a deterministic, OpenCode-independent check.

    This — never AgentResult.claimed_done — is what a LangGraph conditional
    edge should branch on. See docs/verification.md.
    """

    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed
