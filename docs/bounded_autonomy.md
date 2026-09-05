# Bounded Autonomy

## The six things LangGraph must never delegate

From the project brief, and enforced by `src/contracts.py:AgentTask` in
every example from `03` onward:

1. **What task is assigned** — `AgentTask.instruction`, written by the
   calling node, not accumulated ambiently from state.
2. **What context is provided** — `AgentTask.context`, a small explicit
   dict (see `context_management.md`).
3. **What workspace is available** — `AgentTask.workspace`, a `Path` the
   caller creates and owns (see `sandboxing.md` — and its callout that this
   guarantee is currently backend-dependent: `SDKBackend` cannot enforce it
   by itself the way `CLIBackend` does).
4. **What resources/permissions are allowed** — `AgentTask.permissions`, a
   plain string profile (`"read_only"` / `"read_write"` / `"read_write_exec"`)
   the adapter maps onto real flags/env for the backend in use.
5. **Retry and budget limits** — `AgentTask.max_turns`, `AgentTask.timeout_seconds`.
6. **What counts as success, and what happens next** — never decided inside
   the OpenCode node. Always `src/verifier.py`, always a LangGraph
   conditional edge. See `verification.md`.

If a node needs to loosen one of these per-call, that's fine — it's still a
value the *calling node* chose and put in an `AgentTask`, not a capability
OpenCode reached out and took for itself.

## What "bounded" does not mean

It does not mean micromanaged. Inside the boundary OpenCode gets full
latitude: it can read any file in its workspace, run any shell command its
permission profile allows, take up to `max_turns` internal turns, and change
its own plan as many times as it wants. The bound is on the *edges* of the
task, not on the reasoning inside it. Compare:

- **Over-bounded (an anti-pattern this repo avoids):** LangGraph tells
  OpenCode exactly which file to edit and what diff to apply. At that point
  you've reimplemented the patch engine yourself and OpenCode is a very
  expensive way to call `git apply`.
- **Under-bounded (the failure mode this repo guards against):** OpenCode
  is handed the whole repo, no timeout, no turn limit, and told "make the
  tests pass" with no verifier checking the result. This is close to what
  Deep Agents does by default — a `recursion_limit` of 9,999
  (`research/project_comparison.md`) — which is a reasonable choice *for a
  general-purpose agent product*, but is exactly the default this repo
  argues against for a bounded worker node inside a larger deterministic
  system.
- **Bounded (what every example from `04` on does):** a specific workspace,
  a specific instruction, a permission profile, a turn/time budget, and an
  external verifier that gets the final word.

## Precedent for narrow escalation, not narrow tasks

Open SWE (`research/project_comparison.md`) doesn't bound its Programmer
agent by giving it a tiny task — it gives it a whole repo and lets it
investigate freely. What it bounds is the *exit*: a plan-approval gate
before code gets written, and a hard human-approval requirement specifically
for CI/workflow-file changes, layered on top of an otherwise autonomous
loop. Bounding autonomy is about picking the right control points, not
about making the task itself small. `examples/09` applies the same idea at
workflow scale: the deterministic path stays deterministic, and only a
detected "unusual case" gets escalated to a bounded OpenCode task — the
escalation criterion, not the task content, is what's tightly controlled.

## Permission profiles are a policy, not a promise

`AgentTask.permissions` is advisory to the adapter, which is responsible for
actually enforcing it against whatever backend is running (a real `opencode`
subprocess restricted to a workspace directory, or the fake backend used in
these examples). Real deployments should back this with plain OS-level
enforcement — a restricted, unprivileged OS user for the OpenCode process,
a read-only bind mount for everything outside the workspace — not a
container runtime; see `sandboxing.md` for why this repo treats a
container as an opt-in escalation for a stronger threat model, not the
default. Don't rely on the agent "choosing" to respect a permission
string; the workspace boundary is what actually has to hold.
