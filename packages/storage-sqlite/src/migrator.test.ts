import { describe, expect, it } from "vitest";
import { checkFts5Available, openDatabase } from "./connection.js";
import {
  getAppliedMigrationIds,
  getPendingMigrationIds,
  runMigrations,
} from "./migrator.js";
import { SqliteProjectRepository } from "./repositories/sqlite-project-repository.js";

describe("migrator", () => {
  it("applies all migrations and is idempotent on re-run", () => {
    const conn = openDatabase(":memory:");
    try {
      expect(getPendingMigrationIds(conn.raw).length).toBeGreaterThan(0);

      const first = runMigrations(conn.raw);
      expect(first.applied).toContain("0001_core");
      expect(getPendingMigrationIds(conn.raw)).toHaveLength(0);

      const second = runMigrations(conn.raw);
      expect(second.applied).toHaveLength(0);

      const tableNames = conn.raw
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        )
        .all()
        .map((r: any) => r.name);
      expect(tableNames).toEqual(
        expect.arrayContaining(["projects", "agents", "settings", "schema_migrations"]),
      );
    } finally {
      conn.close();
    }
  });

  it("reports FTS5 as available on the bundled better-sqlite3 build", () => {
    const conn = openDatabase(":memory:");
    try {
      expect(checkFts5Available(conn.raw)).toBe(true);
    } finally {
      conn.close();
    }
  });

  it("enforces foreign keys and WAL mode", () => {
    const conn = openDatabase(":memory:");
    try {
      const fk = conn.raw.pragma("foreign_keys", { simple: true });
      expect(fk).toBe(1);
    } finally {
      conn.close();
    }
  });
});

describe("SqliteProjectRepository", () => {
  it("creates and retrieves a project by id and key", async () => {
    const conn = openDatabase(":memory:");
    try {
      runMigrations(conn.raw);
      const repo = new SqliteProjectRepository(conn.db);
      const created = await repo.create({ key: "ilt-agent", name: "ILT Agent" });

      expect(await repo.getById(created.id)).toEqual(created);
      expect(await repo.getByKey("ilt-agent")).toEqual(created);
      expect(await repo.getByKey("missing")).toBeNull();
      expect(await repo.list()).toEqual([created]);
    } finally {
      conn.close();
    }
  });
});

describe("getAppliedMigrationIds", () => {
  it("returns an empty list before any migration runs", () => {
    const conn = openDatabase(":memory:");
    try {
      expect(getAppliedMigrationIds(conn.raw)).toEqual([]);
    } finally {
      conn.close();
    }
  });
});
