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
- **Open SWE** runs every task inside an isolated sandbox — pluggable
  cloud providers (Daytona, Modal, E2B) or local — tied to a thread id so
  it's resumable. The isolation is the control, not a permission list
  layered on top of a shared filesystem; *which* isolation mechanism backs
  it is a separate, swappable decision. This repo makes a different choice
  than Open SWE's default on that second point — see below.

## What this repo actually uses — and deliberately does not use

`AgentTask.workspace` (`src/contracts.py`) is a `Path` the *calling node*
creates before the task starts:

- **Examples `03`–`07`:** a small fixture directory checked into the repo,
  used read/write within the example's own scope.
- **Example `08`:** a real disposable temp directory, created and destroyed
  by LangGraph nodes, not by OpenCode.

**This guarantee is backend-dependent — read `SDKBackend`'s docstring in
`src/opencode_adapter.py` before relying on it.** `CLIBackend` gets
workspace scoping for free (`subprocess.run(..., cwd=task.workspace)`).
`SDKBackend` does not: the `opencode-ai` Python client (as of
v0.1.0-alpha.36) has no per-session directory parameter at all — a server's
working directory is fixed for that `opencode serve` process's lifetime.
`SDKBackend.execute()` checks the server's actual cwd against
`task.workspace` on every call and fails closed on a mismatch rather than
silently running against the wrong directory, but the operational
consequence is real: using `SDKBackend` with per-task disposable
workspaces (this section's whole point) means running **one `opencode
serve` process per workspace**, not one shared long-lived server. If your
deployment genuinely needs one shared server across many tasks, `CLIBackend`
currently gives you the disposable-workspace property this doc argues for;
`SDKBackend` doesn't, yet.

This repo intentionally does not reach for a container runtime (Docker,
Daytona, or similar) for that isolation, even though several surveyed
projects default to one. A container is a heavier dependency than the
problem needs here: it means a container daemon or cloud sandbox API to
install, configure, and keep working, for tasks whose actual isolation
requirement is "don't let this touch anything outside one directory, and
be able to throw the directory away." A plain OS-level boundary gets that
without the extra moving part:

- A disposable directory (`tempfile.mkdtemp()`, as in example `08`) for
  the default case.
- Where the blast radius of a mistake is real, run the OpenCode process as
  its own unprivileged OS user with write access to nothing but the
  workspace directory (`chown`/`chmod`, or a read-only bind mount for
  everything that isn't the workspace) — no container image, registry, or
  runtime required, just filesystem permissions the OS already enforces.
- Resource limits (a subprocess timeout, `ulimit`-style caps) cover the
  "runaway process" case a container's cgroups would otherwise be used for,
  without adding cgroups as a dependency.

The `AgentTask.workspace` contract doesn't change either way — only what
backs it on disk does. If a real deployment's threat model genuinely needs
stronger isolation than an OS user boundary gives (untrusted multi-tenant
code, for instance), that's a legitimate reason to add a container or a
cloud sandbox provider — but it should be a deliberate response to that
requirement, not the default starting point.

## Wide observation, narrow mutation

A workspace can be generous about what OpenCode can *read* (the whole repo,
for context) while staying narrow about what it can *write* or *execute* —
e.g. a read-write bind mount for one subdirectory, read-only for everything
else. `AgentTask.permissions` (`docs/bounded_autonomy.md`) is the hook for
this; enforcing it is the adapter's and the workspace's job, not something
OpenCode opts into.
