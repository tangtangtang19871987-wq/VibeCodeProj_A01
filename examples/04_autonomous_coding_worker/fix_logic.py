"""Stands in for what OpenCode's own investigate/edit/run/repair loop would
do to broken_project/. See src/opencode_adapter.py's module docstring: this
is a fixture-specific test double, not a general capability, and it lives
in the example directory (not src/) for exactly that reason.

It deliberately takes two tries to get right — a wrong-but-plausible first
fix, then a correction — so that even example 04's *single* bounded
OpenCodeAdapter.run() call visibly shows an internal retry, driven by
FakeBackend re-running `check_command` between attempts.
"""

from pathlib import Path


def simulate_attempt(workspace: Path, attempt: int, last_error: str) -> str:
    converter = workspace / "converter.py"
    text = converter.read_text()
    if "- 32" in text:
        converter.write_text(text.replace("celsius * 9 / 5 - 32", "celsius * 5 / 9 + 32"))
        return "investigated failing tests, tried swapping the conversion ratio"
    if "5 / 9" in text:
        converter.write_text(text.replace("celsius * 5 / 9 + 32", "celsius * 9 / 5 + 32"))
        return "previous fix still failed; corrected the ratio and kept the sign fix"
    return "no further changes needed"
