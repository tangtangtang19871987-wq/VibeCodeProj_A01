"""OpenCode's stand-in for this example: applies one plausible-looking fix
and claims done, WITHOUT re-running the tests itself (no check_command is
given to the adapter in this example — see main.py). This is the point:
`claimed_done=True` here is a claim OpenCode believes, not a fact. It's
still wrong. Only src/verifier.py running the real test suite catches it.
"""

from pathlib import Path


def simulate_attempt(workspace: Path, attempt: int, last_error: str) -> str:
    converter = workspace / "converter.py"
    text = converter.read_text()
    converter.write_text(text.replace("celsius * 9 / 5 - 32", "celsius * 5 / 9 + 32"))
    return "swapped the conversion ratio; looks right, didn't re-run the tests"
