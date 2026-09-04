# Verification

## The core rule

**The autonomous kernel must not define its own success.** OpenCode saying
"done" is a claim, not a fact. `AgentResult.claimed_done` is named that way
deliberately — nothing downstream should read it as ground truth.

```text
OpenCode
   |
verifier
 /    \
PASS  FAIL
```

`src/verifier.py` is where ground truth actually gets decided: run the test
suite, check a lint gate, check a file for an expected pattern, run a
domain-specific check — anything deterministic and external to the agent
that did the work. The result is a `VerificationResult` (`src/contracts.py`),
and it's what a LangGraph conditional edge branches on — never
`AgentResult.claimed_done` directly.

## Precedent: verification as a separate process, not a separate step

The single clearest real-world precedent in the whole survey
(`research/project_comparison.md`) is Open SWE: the agent that writes code
(Programmer) is never the agent that decides if it's good enough. A
structurally **separate** LangGraph graph — Reviewer — does read-only
analysis and gates the PR, informed by objective signals (tests, lint)
first. A "needs follow-up" result loops back to the Planner rather than
being forced closed. That's the shape `examples/05` and `06` copy:
verification is not a step inside the worker's loop, it's an independent
process the worker's output is handed to.

Contrast this with Deep Agents and the OpenHands SDK, both of which leave
verification optional and internal (a `FinishTool`-equivalent call, or an
opt-in `critic`) — workable for a general-purpose coding assistant with a
human watching, but not what you want as the sole gate in an unattended
domain workflow.

## Two things a verifier should never do

- **Never trust `claimed_done` as a shortcut.** Even if OpenCode reports
  success, run the check. The check is cheap; a false positive that reaches
  production isn't.
- **Never let the worker mutate the verification target after the check.**
  Run the verifier on the workspace state that resulted from the task, then
  don't let a later "trust me, it's fine" replace re-running it.

## Feedback, not blind retry

A FAIL is more useful than a boolean — `VerificationResult.reason` and
`.details` should contain enough for a retry to actually improve (failing
test output, the exact assertion that broke, a lint error list).
`examples/06_verifier_feedback_loop` feeds this back into the *same*
OpenCode session as the next instruction rather than starting over from
scratch, and caps the number of attempts via `AgentTask.max_turns`-style
budgeting at the graph level (a separate, LangGraph-owned attempt counter —
see `failure_recovery.md`).

## Composing verifiers

`src/verifier.py` provides small deterministic checks (`run_pytest`,
`run_command`, `file_contains`) plus `all_of(*checks)` to combine them. Keep
verifiers boring on purpose: a verifier that itself needs an LLM to
interpret its result has just moved the "who decides success" problem
somewhere less visible, not solved it. If a check genuinely requires
judgment an LLM is good at (e.g. "does this summary read naturally?"), that
LLM call should be its own ordinary LLM node with a narrow, explicit
prompt — not folded into `verifier.py` as if it were deterministic.
