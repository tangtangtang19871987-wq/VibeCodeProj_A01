# 06 — Verifier Feedback Loop

```text
        +-----------------------------+
        v                             |
   OpenCode task --> verifier --FAIL--+ (feed reason back, attempt += 1)
        |
      PASS -> done                     attempt == max_attempts -> give up
```

Builds directly on `05`: same fixture, same wrong-first fix, but a FAIL now
loops back instead of ending the graph. `prepare_retry` is a plain
deterministic node — it takes the verifier's own failure text and folds it
into the next instruction, so the second OpenCode call isn't starting from
nothing.

Two counters matter here, and they are not the same thing
(`docs/failure_recovery.md`): `AgentTask.max_turns` bounds *one* OpenCode
call's own internal iteration (here, deliberately set to 1, so all the
retrying you see is happening at the LangGraph level); `MAX_ATTEMPTS`
bounds the *outer* LangGraph loop across separate OpenCode calls. Watch the
printed attempt log — attempt 1 fails, its exact reason gets folded into
attempt 2's instruction, attempt 2 passes.

## Run

```bash
python examples/06_verifier_feedback_loop/main.py
```

## Read next

`examples/07_context_isolation`.
