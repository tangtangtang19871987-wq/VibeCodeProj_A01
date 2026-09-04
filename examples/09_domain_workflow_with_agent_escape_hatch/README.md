# 09 — Domain Workflow With an Autonomous Escape Hatch

```text
deterministic domain workflow
        |
   unusual problem?
      /      \
    no        yes
    |          |
 continue   OpenCode
              |
           verifier
              |
           continue
```

The capstone. A small order-intake pipeline (parse -> validate -> total)
processes two orders:

- **well-formed** — never comes near OpenCode. `classify` validates it,
  routes straight to `finish_order`. This is the 95% case in a real system,
  and it stays exactly as deterministic as `examples/00`.
- **unusual-shape** — uses the wrong key (`products` instead of `items`)
  and string values with a currency symbol (`"$12.50"`). This isn't a
  one-off `elif` waiting to be written; it's the kind of shape a schema
  validator wasn't built to anticipate. `classify` detects that the order
  fails validation and routes to `escape_hatch_opencode` — the *only* node
  in this entire workflow that touches OpenCode.

After the escape hatch runs, `verify_repair` calls the exact same
`validate_order` function `classify` used — not a rubber stamp, a real
re-check (`docs/verification.md`) — before the (possibly repaired) order is
allowed to continue to `finish_order`. If the repair still doesn't validate,
the order goes to `manual_review` instead of being forced through.

Both orders end up with the same total, computed by the same deterministic
`compute_total` — the escape hatch changed *how the order got into valid
shape*, not what happens to a valid order afterward.

## Why this is the point of the whole repository

Every other example in this repo demonstrates one piece in isolation: a
node calling OpenCode (`03`), an independent verifier (`05`), a retry loop
(`06`), context isolation (`07`), a disposable workspace (`08`). This
example is what it looks like when those pieces are all just... quietly
part of a normal-looking business workflow, invoked in exactly one narrow
place, doing exactly one bounded thing, checked before anything downstream
trusts it. See `docs/engineering_tradeoffs.md`'s "agent first, then
harden" section for how the boundary between the deterministic and
escape-hatch sides of `classify` is expected to move over time, as more
"unusual" shapes turn out to be common enough to deserve their own
deterministic rule.

## Run

```bash
python examples/09_domain_workflow_with_agent_escape_hatch/main.py
```

## Read next

You've finished the example progression. `docs/engineering_tradeoffs.md`
and `research/project_comparison.md` are worth a second read now that
you've seen the pattern end to end.
