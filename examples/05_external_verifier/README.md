# 05 — External Deterministic Verification

```text
OpenCode
   |
verifier
 /    \
PASS  FAIL
```

The fixture is deliberately rigged so OpenCode's fix (swap the ratio) looks
plausible but is still wrong, and OpenCode never re-checks its own work
before claiming done. `src/verifier.py:run_pytest` does the checking that
OpenCode skipped, and the graph's conditional edge branches on *that*
result — not on `AgentResult.claimed_done`. See `docs/verification.md`.

Run it and you should see `FAIL`, with the log making it obvious OpenCode
was confident and wrong. That gap between "the agent thinks it's done" and
"an independent check agrees" is the entire reason `src/verifier.py` exists
as a module separate from `src/opencode_adapter.py`.

Both branches (`report_pass`, `report_fail`) are terminal in this example —
there's no retry yet. That's `06`.

## Run

```bash
python examples/05_external_verifier/main.py
```

## Read next

`examples/06_verifier_feedback_loop`.
