"""Shared infrastructure for the LangGraph + OpenCode teaching examples.

Kept deliberately small — see docs/architecture.md. Three modules:

- contracts: the data shapes that cross the LangGraph <-> OpenCode boundary.
- opencode_adapter: the one place that knows how to invoke OpenCode.
- verifier: deterministic, OpenCode-independent success checks.
"""
