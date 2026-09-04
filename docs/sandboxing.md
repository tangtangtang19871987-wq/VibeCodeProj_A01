# Sandboxing

## Prefer disposable environments over a custom permission system

The brief's principle: "prefer environment isolation over building a large
custom permission system." Concretely, that means: instead of writing code
that tries to police *what* OpenCode does (parsing every shell command it
wants to run, maintaining an allowlist of safe operations, etc.), give it a
disposable place to do it in, and decide afterward — via verification —
whether the result is worth keeping.

```text
create workspace -> hand to OpenCode -> verify -> promote or discard
```

`examples/08_disposable_workspace` implements exactly this lifecycle:
`tempfile.mkdtemp()` for the workspace, the OpenCode node works inside it
with no awareness of anything outside, `src/verifier.py` checks the result
in place, and only on PASS does a deterministic node copy the relevant
files out to a permanent location. On FAIL, the temp directory is removed.
OpenCode never had the ability to touch anything outside that directory in
the first place — the boundary is enforced by *what exists on disk*, not by
OpenCode choosing to behave.

## Why not a custom permission system

Every project surveyed already solved a piece of this, and none of them
did it by carefully enumerating allowed operations at the tool-call level
as the primary defense:

- **OpenCode** (`research/project_comparison.md`) already ships named
  permission tiers (`build` = full access, `plan` = read-only) and, in
  non-interactive mode, auto-denies prompts it can't get a human answer to
  — fail closed. Reuse that instead of building a parallel system.
- **Deep Agents' `SandboxBackendProtocol`**
  (`libs/deepagents/deepagents/middleware/filesystem.py`) only adds an
  `execute` tool when the backend itself is something that can safely run
  shell commands — the sandboxing decision is made once, at backend
  selection, not re-litigated per tool call.
- **Open SWE** runs every task inside an isolated sandbox (Daytona, Modal,
  E2B, or local), tied to a thread id so it's resumable — the isolation is
  the control, not a permission list layered on top of a shared filesystem.

## What this repo actually uses

`AgentTask.workspace` (`src/contracts.py`) is a `Path` the *calling node*
creates before the task starts:

- **Examples `03`–`07`:** a small fixture directory checked into the repo,
  used read/write within the example's own scope.
- **Example `08`:** a real disposable temp directory, created and destroyed
  by LangGraph nodes, not by OpenCode.
- **Production guidance (not implemented here, to keep the repo teaching-sized):**
  the same `workspace` field should point at a container mount or a
  restricted-user chroot when the blast radius of a mistake is real — the
  contract doesn't change, only what backs `workspace` on disk does.

## Wide observation, narrow mutation

A workspace can be generous about what OpenCode can *read* (the whole repo,
for context) while staying narrow about what it can *write* or *execute* —
e.g. a read-write bind mount for one subdirectory, read-only for everything
else. `AgentTask.permissions` (`docs/bounded_autonomy.md`) is the hook for
this; enforcing it is the adapter's and the workspace's job, not something
OpenCode opts into.
