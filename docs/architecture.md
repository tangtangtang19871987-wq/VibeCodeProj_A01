# Architecture

## The shape

```text
                         LangGraph
                 deterministic control plane
                             |
            -----------------------------------
            |                |                |
     deterministic        LLM node        OpenCode node
      domain logic     (ordinary call)   (autonomous kernel)
            |                |                |
            -----------------------------------
                             |
                       verification
                             |
                       next graph step
```

LangGraph is the **only** thing that decides what runs next. Every node —
deterministic Python, a single LLM call, or an OpenCode-backed autonomous
worker — is just a function that takes state in and returns state (or a
`Command`) out. The graph doesn't care what happens *inside* a node; it only
cares about the contract at the edges.

That asymmetry is the whole architecture:

- **Inside an OpenCode node**, OpenCode is free. It can read files, run
  shell commands, fail, retry, change its mind, and take as many internal
  turns as its own loop needs. This repo does not observe or control that
  process step-by-step — see `research/project_comparison.md` for why
  reimplementing that loop would be pure duplication.
- **Outside the node**, none of that freedom exists. LangGraph decides what
  task gets assigned (`src/contracts.py:AgentTask`), what workspace it runs
  in (`docs/sandboxing.md`), how many attempts it gets
  (`docs/failure_recovery.md`), what counts as done
  (`docs/verification.md`), and what happens next.

## Why not just let OpenCode run everything?

Because then you'd have no deterministic control plane at all — no
retryable checkpoints, no independent success criteria, no way to bound
blast radius, no way to run a cheap deterministic path for the 95% of cases
that don't need an autonomous agent. You'd be building (badly, and without
LangGraph's checkpointing/observability) the exact kind of monolithic agent
loop that OpenHands moved *away* from when it split V0 into the modular V1
SDK (`research/project_comparison.md`, OpenHands section).

## Why not just build a bigger LangGraph agent loop ourselves?

Because LangGraph has no opinion about tool use, shell execution, patch
application, or traceback-driven repair — and building a good version of
those is most of the engineering effort in *every* project surveyed
(OpenCode, Deep Agents, the OpenHands SDK, Open SWE). Example `02` in this
repo builds a deliberately tiny agent loop from scratch purely so you can
see what a harness actually does; it is explicitly not meant to be grown
into a real one. Section 3 of `LEARNING_PATH.md` says this again at the
point you'd be tempted to extend it.

## Node types

| Node type | What decides its behavior | Example |
|---|---|---|
| Deterministic domain node | Your code | `examples/00_basic_langgraph`, the non-escape-hatch path in `09` |
| LLM node | A single prompt/response, no loop | `examples/01_llm_inside_graph` |
| Toy agent loop node | A hand-rolled `LLM -> tool -> observation -> LLM` cycle, kept intentionally small | `examples/02_minimal_agent_loop` |
| OpenCode node | OpenCode's own internal loop, bounded by `AgentTask` | `examples/03` onward |
| Verifier node | A deterministic check, external to whichever node produced the result | `examples/05` onward |

## The contract between LangGraph and OpenCode

Defined once, in `src/contracts.py`, and used by every example from `03`
onward:

- `AgentTask` — what LangGraph hands to OpenCode: an instruction, a
  workspace path, a compact context dict, a permission profile, and a
  budget (max turns / timeout). This is the *entire* interface OpenCode
  sees; it cannot ask for more.
- `AgentResult` — what OpenCode hands back: a summary, a pointer to its
  full log (not the log itself — see `context_management.md`), and its own
  claim about whether it finished. That claim is named `claimed_done`, not
  `success`, on purpose: see `verification.md` for why LangGraph never
  trusts it directly.
- `VerificationResult` — produced by `src/verifier.py`, never by the
  OpenCode node itself. This is what LangGraph's conditional edges actually
  branch on.

## Where this pattern breaks down

Documented honestly in `docs/engineering_tradeoffs.md`: latency (every
OpenCode call is a subprocess with its own model round-trips), cost
(bounded but non-trivial autonomy still burns tokens), and tasks that are
genuinely deterministic once understood, where routing to an autonomous
kernel at all is overkill from the start.
