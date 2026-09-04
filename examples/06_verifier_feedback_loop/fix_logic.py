"""Same idea as 04's and 05's fix_logic.py, but written to be called across
*separate* OpenCodeAdapter.run() invocations (main.py passes max_turns=1,
so each call represents exactly one OpenCode "turn"). It decides what to do
by inspecting the current file content on disk, which is what lets it pick
up where the previous attempt left off without any extra plumbing between
LangGraph and the fake backend.
"""

from pathlib import Path


def simulate_attempt(workspace: Path, attempt: int, last_error: str) -> str:
    converter = workspace / "converter.py"
    text = converter.read_text()
    if "- 32" in text:
        converter.write_text(text.replace("celsius * 9 / 5 - 32", "celsius * 5 / 9 + 32"))
        return "first attempt: swapped the conversion ratio"
    if "5 / 9" in text:
        converter.write_text(text.replace("celsius * 5 / 9 + 32", "celsius * 9 / 5 + 32"))
        return "read the verifier's failure output, corrected the ratio this time"
    return "no further changes needed"
