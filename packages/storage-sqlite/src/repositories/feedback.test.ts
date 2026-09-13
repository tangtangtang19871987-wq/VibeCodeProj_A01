import { FeedbackService } from "@lams/application";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { openDatabase, type SqliteConnection } from "../connection.js";
import { runMigrations } from "../migrator.js";
import { SqliteDeliveryRepository } from "./sqlite-recall-repository.js";
import { SqliteFeedbackRepository } from "./sqlite-feedback-repository.js";
import { SqliteProjectRepository } from "./sqlite-project-repository.js";
import { SqliteSessionRepository } from "./sqlite-session-repository.js";
import { SqliteMemoryRepository } from "./sqlite-memory-repository.js";
import { humanActor } from "@lams/domain";

describe("FeedbackService", () => {
  let conn: SqliteConnection;

  beforeEach(() => {
    conn = openDatabase(":memory:");
    runMigrations(conn.raw);
  });

  afterEach(() => {
    conn.close();
  });

  it("always records feedback as 'reported' evidence, never 'observed'", async () => {
    const feedbackRepo = new SqliteFeedbackRepository(conn.db);
    const deliveryRepo = new SqliteDeliveryRepository(conn.db);
    const service = new FeedbackService(feedbackRepo, deliveryRepo);

    const projects = new SqliteProjectRepository(conn.db);
    const sessions = new SqliteSessionRepository(conn.db);
    const memories = new SqliteMemoryRepository(conn.db);

    const project = await projects.create({ key: "p1", name: "P1" });
    const session = await sessions.open({ harness: "test", projectId: project.id });
    const { memory } = await memories.createWithFirstVersion({
      kind: "fact",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: humanActor("dev-1"),
      version: { title: "t", content: "c" },
    });

    const record = await service.record({
      sessionId: session.id,
      memoryId: memory.id,
      eventType: "useful",
    });

    expect(record.evidenceLevel).toBe("reported");

    const delivered = await deliveryRepo.listForSession(session.id);
    expect(delivered).toHaveLength(1);
    expect(delivered[0]!.stage).toBe("used");
    expect(delivered[0]!.evidenceLevel).toBe("reported");

    const forMemory = await service.listForMemory(memory.id);
    expect(forMemory).toHaveLength(1);
    expect(forMemory[0]!.eventType).toBe("useful");
  });

  it("maps injected/outcome_link feedback to the correct delivery stage", async () => {
    const feedbackRepo = new SqliteFeedbackRepository(conn.db);
    const deliveryRepo = new SqliteDeliveryRepository(conn.db);
    const service = new FeedbackService(feedbackRepo, deliveryRepo);
    const projects = new SqliteProjectRepository(conn.db);
    const sessions = new SqliteSessionRepository(conn.db);
    const memories = new SqliteMemoryRepository(conn.db);

    const project = await projects.create({ key: "p2", name: "P2" });
    const session = await sessions.open({ harness: "test", projectId: project.id });
    const { memory } = await memories.createWithFirstVersion({
      kind: "fact",
      scope: { level: "project", id: project.id },
      status: "verified",
      createdBy: humanActor("dev-1"),
      version: { title: "t", content: "c" },
    });

    await service.record({ sessionId: session.id, memoryId: memory.id, eventType: "injected" });
    await service.record({
      sessionId: session.id,
      memoryId: memory.id,
      eventType: "outcome_link",
      outcomeMetric: { success: 1 },
    });

    const delivered = await deliveryRepo.listForSession(session.id);
    expect(delivered.map((d) => d.stage)).toEqual(["injected", "outcome_linked"]);
  });
});
