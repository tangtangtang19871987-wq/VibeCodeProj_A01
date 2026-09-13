import { humanActor, agentActor } from "@lams/domain";
import { MemoryService } from "@lams/application";
import { beforeEach, afterEach, describe, expect, it } from "vitest";
import { openDatabase, type SqliteConnection } from "../connection.js";
import { runMigrations } from "../migrator.js";
import { SqliteMemoryRepository } from "./sqlite-memory-repository.js";
import {
  SqliteMemoryRelationRepository,
  SqliteProvenanceRepository,
  SqliteReviewRepository,
} from "./sqlite-provenance-review-repository.js";
import { SqliteFtsSearchAdapter } from "./sqlite-fts-search.js";

function makeService(conn: SqliteConnection) {
  const memories = new SqliteMemoryRepository(conn.db);
  const provenance = new SqliteProvenanceRepository(conn.db);
  const reviews = new SqliteReviewRepository(conn.db);
  const relations = new SqliteMemoryRelationRepository(conn.db);
  const fts = new SqliteFtsSearchAdapter(conn);
  return { service: new MemoryService(memories, provenance, reviews, relations, fts), memories };
}

describe("MemoryService + SQLite repositories", () => {
  let conn: SqliteConnection;

  beforeEach(() => {
    conn = openDatabase(":memory:");
    runMigrations(conn.raw);
  });

  afterEach(() => {
    conn.close();
  });

  it("creates a memory with an immutable first version, defaulting agents to draft", async () => {
    const { service } = makeService(conn);
    const { memory, version } = await service.create({
      kind: "lesson",
      scope: { level: "project", id: "ilt-agent" },
      status: "verified", // should be ignored/forced to draft for agent actors
      createdBy: agentActor("opencode-1"),
      version: { title: "LBFGS warm start", content: "Use Adam warm start then LBFGS." },
    });

    expect(memory.status).toBe("draft");
    expect(version.version).toBe(1);
    expect(memory.currentVersionId).toBe(version.id);
  });

  it("allows a human to create a memory as verified directly", async () => {
    const { service } = makeService(conn);
    const { memory } = await service.create({
      kind: "decision",
      scope: { level: "project", id: "p1" },
      status: "verified",
      createdBy: humanActor("dev-1"),
      version: { title: "Use LBFGS", content: "Decision: use LBFGS for dense contact." },
    });
    expect(memory.status).toBe("verified");
  });

  it("editing creates a new immutable version and logs a review action", async () => {
    const { service } = makeService(conn);
    const actor = humanActor("dev-1");
    const { memory } = await service.create({
      kind: "fact",
      scope: { level: "project", id: "p1" },
      createdBy: actor,
      version: { title: "v1", content: "original content" },
    });

    const { version: v2 } = await service.addVersion(memory.id, {
      title: "v1 (edited)",
      content: "corrected content",
      changeReason: "fixing a typo",
      createdBy: actor,
    });

    expect(v2.version).toBe(2);
    const detail = await service.getDetail(memory.id);
    expect(detail!.versions).toHaveLength(2);
    expect(detail!.versions[0]!.content).toBe("original content");
    expect(detail!.currentVersion.content).toBe("corrected content");
    expect(detail!.reviews.some((r) => r.action === "edit")).toBe(true);
  });

  it("enforces valid status transitions and rejects invalid ones", async () => {
    const { service } = makeService(conn);
    const actor = humanActor("dev-1");
    const { memory } = await service.create({
      kind: "fact",
      scope: { level: "project", id: "p1" },
      createdBy: actor,
      version: { title: "t", content: "c" },
    });

    await service.approve(memory.id, actor);
    const afterApprove = await service.getDetail(memory.id);
    expect(afterApprove!.memory.status).toBe("verified");

    await expect(service.approve(memory.id, actor)).resolves.toBeDefined(); // no-op transition to same status

    await service.reject(memory.id, actor, "actually wrong");
    // Now the memory is rejected; approve() should fail (rejected -> verified not allowed)
    await expect(service.approve(memory.id, actor)).rejects.toThrow();
  });

  it("excludes rejected and superseded memories from full-text search by default", async () => {
    const { service } = makeService(conn);
    const actor = humanActor("dev-1");

    const older = await service.create({
      kind: "lesson",
      scope: { level: "project", id: "p1" },
      status: "verified",
      createdBy: actor,
      version: { title: "optimizer choice", content: "Always use Adam optimizer." },
    });

    const newer = await service.create({
      kind: "lesson",
      scope: { level: "project", id: "p1" },
      status: "verified",
      createdBy: actor,
      version: {
        title: "optimizer choice v2",
        content: "For dense contact cases, warm start with Adam then switch to LBFGS optimizer.",
      },
    });

    await service.supersede(older.memory.id, newer.memory.id, actor, "superseded by refined guidance");

    const results = await service.search("optimizer", {});
    const ids = results.map((r) => r.memory.id);
    expect(ids).toContain(newer.memory.id);
    expect(ids).not.toContain(older.memory.id);

    const detail = await service.getDetail(older.memory.id);
    expect(detail!.memory.status).toBe("superseded");
    expect(detail!.memory.supersededById).toBe(newer.memory.id);
    expect(detail!.relations.some((r) => r.type === "supersedes")).toBe(true);
  });

  it("keeps a rejected memory out of search but still readable by ID (audit trail)", async () => {
    const { service } = makeService(conn);
    const actor = humanActor("dev-1");
    const { memory } = await service.create({
      kind: "fact",
      scope: { level: "project", id: "p1" },
      createdBy: actor,
      version: { title: "wrong fact", content: "The unique widget frobnicator value is 42." },
    });
    await service.reject(memory.id, actor, "incorrect");

    const results = await service.search("frobnicator", {});
    expect(results).toHaveLength(0);

    const detail = await service.getDetail(memory.id);
    expect(detail!.memory.status).toBe("rejected");
    expect(detail!.reviews.some((r) => r.action === "reject")).toBe(true);
  });

  it("tombstones a memory so it is excluded by default but still fully readable", async () => {
    const { service } = makeService(conn);
    const actor = humanActor("dev-1");
    const { memory } = await service.create({
      kind: "fact",
      scope: { level: "project", id: "p1" },
      createdBy: actor,
      version: { title: "t", content: "gone soon" },
    });

    await service.delete(memory.id, actor);

    const page = await service.list({});
    expect(page.items.some((i) => i.memory.id === memory.id)).toBe(false);

    const pageWithDeleted = await service.list({ includeDeleted: true });
    expect(pageWithDeleted.items.some((i) => i.memory.id === memory.id)).toBe(true);

    const detail = await service.getDetail(memory.id);
    expect(detail!.memory.status).toBe("deleted");
    expect(detail!.currentVersion.content).toBe("gone soon");
  });

  it("links provenance to a memory", async () => {
    const { service } = makeService(conn);
    const actor = agentActor("opencode-1");
    const { memory } = await service.create({
      kind: "episode",
      scope: { level: "session", id: "sess-1" },
      createdBy: actor,
      version: { title: "tried Adam", content: "Adam diverged after 200 steps." },
    });

    await service.addProvenance({
      memoryId: memory.id,
      sourceType: "session_event",
      sourceRef: "evt_123",
      sessionId: "sess-1",
      excerpt: "loss diverged at step 214",
    });

    const detail = await service.getDetail(memory.id);
    expect(detail!.provenance).toHaveLength(1);
    expect(detail!.provenance[0]!.sourceRef).toBe("evt_123");
  });

  it("survives a reopened connection to the same file (restart persistence)", async () => {
    const { promises: fs } = await import("node:fs");
    const os = await import("node:os");
    const path = await import("node:path");
    const dbPath = path.join(await fs.mkdtemp(path.join(os.tmpdir(), "lams-test-")), "db.sqlite3");

    const conn1 = openDatabase(dbPath);
    runMigrations(conn1.raw);
    const svc1 = makeService(conn1).service;
    const { memory } = await svc1.create({
      kind: "fact",
      scope: { level: "project", id: "p1" },
      createdBy: humanActor("dev-1"),
      version: { title: "persisted", content: "should survive restart" },
    });
    conn1.close();

    const conn2 = openDatabase(dbPath);
    const svc2 = makeService(conn2).service;
    const detail = await svc2.getDetail(memory.id);
    expect(detail?.currentVersion.content).toBe("should survive restart");
    conn2.close();
  });
});
