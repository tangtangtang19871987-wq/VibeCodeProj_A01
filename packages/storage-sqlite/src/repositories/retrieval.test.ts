import { MemoryService, RetrievalService } from "@lams/application";
import { humanActor } from "@lams/domain";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { openDatabase, type SqliteConnection } from "../connection.js";
import { runMigrations } from "../migrator.js";
import { SqliteFtsSearchAdapter } from "./sqlite-fts-search.js";
import { SqliteMemoryRepository } from "./sqlite-memory-repository.js";
import {
  SqliteMemoryRelationRepository,
  SqliteProvenanceRepository,
  SqliteReviewRepository,
} from "./sqlite-provenance-review-repository.js";
import { SqliteDeliveryRepository, SqliteRecallRepository } from "./sqlite-recall-repository.js";
import { SqliteProjectRepository } from "./sqlite-project-repository.js";
import { SqliteSessionRepository } from "./sqlite-session-repository.js";

function makeServices(conn: SqliteConnection) {
  const memories = new SqliteMemoryRepository(conn.db);
  const provenance = new SqliteProvenanceRepository(conn.db);
  const reviews = new SqliteReviewRepository(conn.db);
  const relations = new SqliteMemoryRelationRepository(conn.db);
  const fts = new SqliteFtsSearchAdapter(conn);
  const memoryService = new MemoryService(memories, provenance, reviews, relations, fts);

  const projects = new SqliteProjectRepository(conn.db);
  const sessions = new SqliteSessionRepository(conn.db);
  const recallRepo = new SqliteRecallRepository(conn.db);
  const delivery = new SqliteDeliveryRepository(conn.db);

  return { memoryService, projects, sessions, recallRepo, delivery, memories };
}

const actor = humanActor("dev-1");

describe("RetrievalService (golden retrieval tests)", () => {
  let conn: SqliteConnection;

  beforeEach(() => {
    conn = openDatabase(":memory:");
    runMigrations(conn.raw);
  });

  afterEach(() => {
    conn.close();
  });

  async function setupSession(services: ReturnType<typeof makeServices>) {
    const project = await services.projects.create({ key: "p1", name: "P1" });
    const session = await services.sessions.open({
      harness: "test-harness",
      projectId: project.id,
      agentName: "test-agent",
    });
    return { project, session };
  }

  it("returns a verified memory ranked above nothing when it's the only match, and persists a complete trace", async () => {
    const services = makeServices(conn);
    const { project, session } = await setupSession(services);

    const { memory } = await services.memoryService.create({
      kind: "lesson",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: actor,
      version: { title: "LBFGS", content: "Use LBFGS after a short Adam warm start." },
    });

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const { packet, trace } = await retrieval.recall({ sessionId: session.id, query: "LBFGS warm start" });

    expect(packet.memories).toHaveLength(1);
    expect(packet.memories[0]!.id).toBe(memory.id);
    expect(packet.memories[0]!.why.length).toBeGreaterThan(0);
    expect(trace.candidateCount).toBe(1);
    expect(trace.returnedCount).toBe(1);
    expect(trace.strategy).toBe("deterministic-fts-v1");

    const persisted = await services.recallRepo.getTrace(trace.id);
    expect(persisted).toEqual(trace);
    const candidates = await services.recallRepo.listCandidates(trace.id);
    expect(candidates).toHaveLength(1);
    expect(candidates[0]!.selected).toBe(true);
    expect(candidates[0]!.returned).toBe(true);
  });

  it("excludes a superseded memory from recall even when it matches the query text", async () => {
    const services = makeServices(conn);
    const { project, session } = await setupSession(services);

    const older = await services.memoryService.create({
      kind: "lesson",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: actor,
      version: { title: "old", content: "Always use Adam optimizer for every case." },
    });
    const newer = await services.memoryService.create({
      kind: "lesson",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: actor,
      version: { title: "new", content: "Use Adam optimizer only as a warm start, then switch." },
    });
    await services.memoryService.supersede(older.memory.id, newer.memory.id, actor);

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const { packet, trace } = await retrieval.recall({ sessionId: session.id, query: "Adam optimizer" });

    expect(packet.memories.map((m) => m.id)).toEqual([newer.memory.id]);
    const candidates = await services.recallRepo.listCandidates(trace.id);
    const oldCandidate = candidates.find((c) => c.memoryId === older.memory.id);
    expect(oldCandidate?.eligible).toBe(false);
    expect(oldCandidate?.exclusionReasons).toContain("SUPERSEDED_EXCLUDED");
  });

  it("excludes drafts by default and includes them when includeDrafts is set", async () => {
    const services = makeServices(conn);
    const { project, session } = await setupSession(services);

    const { memory } = await services.memoryService.create({
      kind: "warning",
      scope: { level: "project", id: project.id },
      createdBy: actor, // defaults to draft
      version: { title: "mesh warning", content: "Refining the mesh past 0.5um can silently diverge." },
    });

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const withoutDrafts = await retrieval.recall({ sessionId: session.id, query: "mesh diverge" });
    expect(withoutDrafts.packet.memories).toHaveLength(0);
    const candidatesWithout = await services.recallRepo.listCandidates(withoutDrafts.trace.id);
    expect(candidatesWithout[0]?.exclusionReasons).toContain("DRAFT_EXCLUDED_BY_POLICY");

    const withDrafts = await retrieval.recall({
      sessionId: session.id,
      query: "mesh diverge",
      includeDrafts: true,
    });
    expect(withDrafts.packet.memories.map((m) => m.id)).toEqual([memory.id]);
  });

  it("returns a valid empty packet (with a persisted trace) when nothing matches", async () => {
    const services = makeServices(conn);
    const { session } = await setupSession(services);

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const { packet, trace } = await retrieval.recall({ sessionId: session.id, query: "nonexistent gibberish query" });
    expect(packet.memories).toEqual([]);
    expect(packet.budget.returnedItems).toBe(0);
    expect(trace.candidateCount).toBe(0);
    const persisted = await services.recallRepo.getTrace(trace.id);
    expect(persisted).not.toBeNull();
  });

  it("never returns more characters than the requested budget, even with many eligible candidates", async () => {
    const services = makeServices(conn);
    const { project, session } = await setupSession(services);

    for (let i = 0; i < 5; i++) {
      await services.memoryService.create({
        kind: "fact",
        scope: { level: "project", id: project.id },
        status: "verified",
        createdBy: actor,
        version: {
          title: `budget fact ${i}`,
          content: `Budget test content number ${i} repeated `.repeat(10),
        },
      });
    }

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const { packet, trace } = await retrieval.recall({
      sessionId: session.id,
      query: "budget test content",
      maxItems: 10,
      maxChars: 300,
    });

    const totalChars = packet.memories.reduce((sum, m) => sum + m.content.length, 0);
    expect(totalChars).toBeLessThanOrEqual(300);
    expect(trace.returnedChars).toBeLessThanOrEqual(300);
    expect(packet.memories.length).toBeLessThan(5);

    const candidates = await services.recallRepo.listCandidates(trace.id);
    expect(candidates.some((c) => c.exclusionReasons.includes("BUDGET_EXCLUDED"))).toBe(true);
  });

  it("preserves historical content: a later edit does not change what a past recall's candidate points to", async () => {
    const services = makeServices(conn);
    const { project, session } = await setupSession(services);

    const { memory } = await services.memoryService.create({
      kind: "fact",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: actor,
      version: { title: "v1", content: "The unique frobnicator threshold is 42." },
    });

    const retrieval = new RetrievalService(
      services.memories,
      new SqliteProvenanceRepository(conn.db),
      new SqliteFtsSearchAdapter(conn),
      services.sessions,
      services.recallRepo,
      services.delivery,
    );

    const first = await retrieval.recall({ sessionId: session.id, query: "frobnicator threshold" });
    expect(first.packet.memories[0]!.content).toBe("The unique frobnicator threshold is 42.");

    await services.memoryService.addVersion(memory.id, {
      title: "v2",
      content: "The unique frobnicator threshold is 99 (corrected).",
      createdBy: actor,
    });

    const candidates = await services.recallRepo.listCandidates(first.trace.id);
    const historicalVersion = await services.memories.getVersion(candidates[0]!.memoryVersionId);
    expect(historicalVersion?.content).toBe("The unique frobnicator threshold is 42.");

    const second = await retrieval.recall({ sessionId: session.id, query: "frobnicator threshold" });
    expect(second.packet.memories[0]!.content).toBe("The unique frobnicator threshold is 99 (corrected).");
  });
});
