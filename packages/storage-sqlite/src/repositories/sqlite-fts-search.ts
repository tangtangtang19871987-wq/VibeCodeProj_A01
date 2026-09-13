import type { FtsSearchPort, SearchCandidate } from "@lams/application";
import type { SqliteConnection } from "../connection.js";

/**
 * Escapes an arbitrary user query into a safe FTS5 MATCH expression: quote
 * every token so characters like `-`, `"`, `*`, `(` never get parsed as
 * FTS5 query syntax (Section 18.4 — memory content, and by extension the
 * query itself, is data, never trusted syntax).
 */
function toFtsQuery(query: string): string {
  const tokens = query
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => `"${t.replace(/"/g, '""')}"*`);
  return tokens.length > 0 ? tokens.join(" ") : '""';
}

export class SqliteFtsSearchAdapter implements FtsSearchPort {
  constructor(private readonly conn: SqliteConnection) {}

  async search(query: string, limit: number): Promise<SearchCandidate[]> {
    const ftsQuery = toFtsQuery(query);
    // Only the memory's *current* version is searchable — older versions
    // stay indexed (so historical recall reconstruction can still work
    // later) but must not surface in an ordinary search (Section 5.4, 15.4).
    const rows = this.conn.raw
      .prepare(
        `SELECT mv.memory_id as memoryId, mv.id as memoryVersionId, bm25(memory_fts) as rawScore
         FROM memory_fts
         JOIN memory_versions mv ON mv.rowid = memory_fts.rowid
         JOIN memories m ON m.id = mv.memory_id AND m.current_version_id = mv.id
         WHERE memory_fts MATCH ?
         ORDER BY rawScore ASC
         LIMIT ?`,
      )
      .all(ftsQuery, limit) as SearchCandidate[];
    return rows;
  }
}
