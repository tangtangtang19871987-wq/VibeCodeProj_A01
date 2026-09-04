# Learning Path

A suggested order through this repository, for an experienced Python
engineer who knows how to write code but hasn't necessarily built an agent
system before. Each stop names a question worth being able to answer before
moving on.

## 1. Research first

- Read `research/sources.md` — know what's a primary source (a specific
  file/function fetched from the actual repo) versus general knowledge.
- Read `research/project_comparison.md` in full, including the
  cross-cutting synthesis table at the end.

**Can you answer:** which single project in the comparison puts
verification in a structurally separate place from the agent that did the
work, and why does that matter more than the other four projects'
approaches to the same question?

## 2. The architecture, in the abstract

- `docs/architecture.md`

**Can you answer:** what is the entire interface OpenCode sees from
LangGraph, and what does LangGraph refuse to give it?

## 3. Deterministic ground floor

- `examples/00_basic_langgraph`

**Can you answer:** what would have to change about this example for it to
stop being "deterministic"?

## 4. One LLM call, still no autonomy

- `examples/01_llm_inside_graph`

**Can you answer:** why is a single LLM call inside a graph node not yet
"an agent"?

## 5. The smallest possible agent loop — and why to stop here

- `examples/02_minimal_agent_loop`

Read the docstring at the top of `main.py` twice. This is the one place in
the repository explicitly warning you not to extend what you're looking
at.

**Can you answer:** name three capabilities a real coding-agent loop needs
that this toy loop doesn't have, and which project in `research/` already
has all three.

## 6. OpenCode enters the graph

- `docs/bounded_autonomy.md`
- `examples/03_opencode_as_node`
- `src/contracts.py` (read the whole file — it's short)

**Can you answer:** which six decisions does `AgentTask` take away from
OpenCode, and where in `src/contracts.py` is each one encoded?

## 7. Autonomy, unleashed inside its box

- `examples/04_autonomous_coding_worker`
- `examples/04_autonomous_coding_worker/fix_logic.py`'s docstring

**Can you answer:** how many attempts did the fake backend take to fix the
bug, and how would you find that out by reading the printed session log
rather than the source code?

## 8. Somebody else has to say it's done

- `docs/verification.md`
- `examples/05_external_verifier`

**Can you answer:** in this example, was `AgentResult.claimed_done` true or
false, and did the workflow still report FAIL? Why is that the correct
behavior rather than a bug?

## 9. Failing forward

- `docs/failure_recovery.md`
- `examples/06_verifier_feedback_loop`

**Can you answer:** what are the two different attempt-counters in play in
this example, and what does each one bound?

## 10. What crosses the boundary, and what doesn't

- `docs/context_management.md`
- `examples/07_context_isolation`

**Can you answer:** by what factor did the "leaked" state representation
exceed the "isolated" one when you ran this example, and where does that
gap come from?

## 11. Workspaces you can throw away

- `docs/sandboxing.md`
- `examples/08_disposable_workspace`

**Can you answer:** what decides whether a workspace's contents get
promoted, and is OpenCode ever consulted in that decision?

## 12. The capstone

- `examples/09_domain_workflow_with_agent_escape_hatch`
- `docs/engineering_tradeoffs.md`

**Can you answer:** in the capstone's graph, how many nodes touch
OpenCode, and what has to be true about an order for execution to reach
that node at all?

## 13. Close the loop

- Re-read `research/project_comparison.md`'s "what this repository
  deliberately does NOT rebuild" section.
- Re-read `docs/engineering_tradeoffs.md`.

**Can you answer, in your own words and without re-reading:** for a real
task you actually have in mind, would this architecture help — and if not,
which specific "where it works poorly" bullet in
`docs/engineering_tradeoffs.md` is the reason?
