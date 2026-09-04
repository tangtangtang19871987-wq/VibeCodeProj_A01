# 04 — Autonomous Coding Worker

A tiny broken project (`broken_project/`: a `celsius_to_fahrenheit` with a
sign bug, and two failing tests) is copied into a fresh disposable
workspace and handed to OpenCode as one bounded task: "the tests are
failing, investigate, fix, and re-run until they pass."

Run it and read the printed session log: OpenCode's stand-in tries a wrong
fix first (swaps the conversion ratio instead of the sign), sees the test
still fails, and corrects it on the second internal attempt — all inside a
single `OpenCodeAdapter.run()` call. That inspect/edit/run/repeat cycle is
the capability this whole repository is built around reusing rather than
rebuilding (see `fix_logic.py`'s docstring and
`research/project_comparison.md`).

What's missing on purpose: nothing outside OpenCode checks its work. The
graph trusts `claimed_done`. That's the gap `05` closes.

## Run

```bash
pip install -r requirements.txt   # langgraph + pytest
python examples/04_autonomous_coding_worker/main.py
```

## Read next

`examples/05_external_verifier`.
