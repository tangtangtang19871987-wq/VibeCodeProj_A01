# ADR 0004: Process model and data directory resolution

Status: Accepted

## Context

The daemon must run as one long-lived local process (Section 8.1) and store
its database/artifacts in a predictable, OS-appropriate location (Section
19.1) without requiring configuration.

## Decision

- One Fastify process serves REST, the web UI (in production, a prebuilt
  static bundle — see ADR 0007), and the Streamable HTTP MCP endpoint, all
  bound to `127.0.0.1` by default (Section 18.1).
- Default data directory resolution uses `env-paths` (`env-paths("lams")`)
  to get the OS-conventional application data directory
  (e.g. `~/.local/share/lams` on Linux, `~/Library/Application Support/lams`
  on macOS, `%APPDATA%/lams` on Windows).
- The `LAMS_DATA_DIR` environment variable overrides this unconditionally.
- Inside the data directory: `db.sqlite3` (+ WAL/SHM files), `artifacts/`,
  and `logs/`.
- A `lams` CLI (`apps/daemon/src/cli.ts`) exposes `start`, `stop`, `status`,
  `doctor`, `backup`, `export`, `import`, `mcp-stdio`, matching Section
  19.2. `pnpm dev` runs the daemon in the foreground for development.

## Consequences

A clean checkout with no configuration writes its data to a stable, OS-
correct location and never inside an arbitrary client project directory,
satisfying Section 19.1's requirement.

## Alternatives considered

Storing data under the repository/CWD by default: rejected because it makes
the daemon behave differently depending on where it is launched from, and
risks a database landing inside a client's project repository.
