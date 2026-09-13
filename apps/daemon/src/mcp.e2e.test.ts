import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { buildApp } from "./app.js";
import { createContext, type AppContext } from "./context.js";
import type { LamsConfig } from "./config.js";

/**
 * PRD Section 22.5: start a real daemon on an ephemeral port, connect with
 * a real MCP client, open a session, remember, review via a fixture-style
 * direct service call (standing in for the human review UI), recall from
 * a second session/client, send feedback, and close the session — proving
 * two clients genuinely share one memory universe (Milestone 3 acceptance
 * criteria) and that the MCP surface only ever exposes the five tools.
 */
describe("MCP end-to-end (Streamable HTTP)", () => {
  let ctx: AppContext;
  let baseUrl: string;
  let app: Awaited<ReturnType<typeof buildApp>>;

  beforeAll(async () => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "lams-mcp-e2e-"));
    const config: LamsConfig = {
      dataDir,
      dbPath: ":memory:",
      artifactsDir: path.join(dataDir, "artifacts"),
      logsDir: path.join(dataDir, "logs"),
      host: "127.0.0.1",
      port: 0,
    };
    ctx = createContext(config);
    app = await buildApp(ctx);
    await app.listen({ host: "127.0.0.1", port: 0 });
    const address = app.server.address();
    if (!address || typeof address === "string") throw new Error("unexpected server address");
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  afterAll(async () => {
    await app.close();
    ctx.close();
  });

  async function connectClient(name: string) {
    const client = new Client({ name, version: "0.0.1" });
    const transport = new StreamableHTTPClientTransport(new URL(`${baseUrl}/mcp`));
    await client.connect(transport);
    return client;
  }

  async function callTool(client: Client, name: string, args: Record<string, unknown>) {
    const result = await client.callTool({ name, arguments: args });
    const text = (result.content as { type: string; text: string }[])[0]?.text ?? "{}";
    return { raw: result, data: JSON.parse(text) as any };
  }

  it("exposes exactly the five tools defined in the PRD", async () => {
    const client = await connectClient("tool-list-check");
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual(
      ["memory_feedback", "memory_get", "memory_recall", "memory_remember", "memory_session"].sort(),
    );
    await client.close();
  });

  it("two different clients can open distinct sessions against the same project", async () => {
    const clientA = await connectClient("harness-a");
    const clientB = await connectClient("harness-b");

    const sessionA = await callTool(clientA, "memory_session", {
      action: "open",
      project: "shared-project",
      harness: "harness-a",
      agent_name: "agent-a",
    });
    const sessionB = await callTool(clientB, "memory_session", {
      action: "open",
      project: "shared-project",
      harness: "harness-b",
      agent_name: "agent-b",
    });

    expect(sessionA.data.session_id).toBeTruthy();
    expect(sessionB.data.session_id).toBeTruthy();
    expect(sessionA.data.session_id).not.toBe(sessionB.data.session_id);
    expect(sessionA.data.project_id).toBe(sessionB.data.project_id);

    await clientA.close();
    await clientB.close();
  });

  it("a memory proposed by one client becomes retrievable by another once approved", async () => {
    const clientA = await connectClient("harness-a-2");
    const clientB = await connectClient("harness-b-2");

    const sessionA = await callTool(clientA, "memory_session", {
      action: "open",
      project: "cross-client-project",
      harness: "harness-a",
    });
    const sessionAId = sessionA.data.session_id as string;

    const remembered = await callTool(clientA, "memory_remember", {
      session_id: sessionAId,
      candidates: [
        {
          kind: "lesson",
          title: "Zorbatron calibration",
          content: "Zorbatron units require a 6-hour cryo-soak before first calibration.",
          reasonWorthRemembering: "Discovered after two failed calibration attempts.",
        },
      ],
    });
    const memoryId = remembered.data.candidates[0].memory_id as string;
    expect(remembered.data.candidates[0].status).toBe("draft");

    // Draft memories are excluded from recall by default (Section 7.1) —
    // approve it the way a human would via the Review Queue, using the
    // same MemoryService the REST API and UI use.
    const approved = await ctx.memoryService.approve(memoryId, {
      type: "human",
      id: "reviewer",
      displayName: "Reviewer",
    });
    expect(approved.status).toBe("verified");

    const sessionB = await callTool(clientB, "memory_session", {
      action: "open",
      project: "cross-client-project",
      harness: "harness-b",
    });
    const sessionBId = sessionB.data.session_id as string;

    const recalled = await callTool(clientB, "memory_recall", {
      session_id: sessionBId,
      query: "Zorbatron cryo-soak calibration",
    });
    expect(recalled.data.memories.map((m: any) => m.id)).toContain(memoryId);
    const traceId = recalled.data.traceId as string;

    const got = await callTool(clientB, "memory_get", {
      session_id: sessionBId,
      memory_ids: [memoryId],
      include_provenance: true,
    });
    expect(got.data.memories[0].content).toContain("cryo-soak");

    const feedback = await callTool(clientB, "memory_feedback", {
      session_id: sessionBId,
      trace_id: traceId,
      memory_ids: [memoryId],
      event_type: "useful",
    });
    expect(feedback.data.recorded[0].evidenceLevel).toBe("reported");

    await callTool(clientA, "memory_session", {
      action: "close",
      session_id: sessionAId,
      status: "completed",
      outcome: { result: "success" },
    });
    await callTool(clientB, "memory_session", {
      action: "close",
      session_id: sessionBId,
      status: "completed",
      outcome: { result: "success" },
    });

    // Every MCP recall must appear immediately in the Session/Retrieval
    // Inspector data (Milestone 3 acceptance criteria) — verify via the
    // same repositories the REST API and UI read from.
    const persistedTrace = await ctx.recallRepo.getTrace(traceId);
    expect(persistedTrace).not.toBeNull();
    const deliveryEvents = await ctx.deliveryRepo.listForTrace(traceId);
    expect(deliveryEvents.some((e) => e.stage === "returned")).toBe(true);
    expect(deliveryEvents.some((e) => e.stage === "used")).toBe(true);

    await clientA.close();
    await clientB.close();
  });

  it("rejects malformed tool input with a stable error and writes nothing", async () => {
    const client = await connectClient("malformed-input-check");

    const before = await ctx.projectService.list();

    const result = await client.callTool({
      name: "memory_session",
      // action=open requires project+harness; omit both.
      arguments: { action: "open" },
    });
    expect(result.isError).toBe(true);
    const text = (result.content as { type: string; text: string }[])[0]?.text ?? "";
    expect(text).toContain("VALIDATION_ERROR");

    const after = await ctx.projectService.list();
    expect(after).toEqual(before);

    await client.close();
  });

  it("caps memory_remember at 5 candidates and memory_get at 10 ids (schema-enforced)", async () => {
    const client = await connectClient("limits-check");
    const session = await callTool(client, "memory_session", {
      action: "open",
      project: "limits-project",
      harness: "limits-harness",
    });

    // Array-length limits are enforced by the tool's input schema itself
    // (InvalidParams at the protocol level, surfaced as an error result),
    // before our handler ever runs — no partial writes are possible.
    const tooMany = await client.callTool({
      name: "memory_remember",
      arguments: {
        session_id: session.data.session_id,
        candidates: Array.from({ length: 6 }, (_, i) => ({
          kind: "fact",
          title: `t${i}`,
          content: `c${i}`,
        })),
      },
    });
    expect(tooMany.isError).toBe(true);

    const tooManyIds = await client.callTool({
      name: "memory_get",
      arguments: {
        session_id: session.data.session_id,
        memory_ids: Array.from({ length: 11 }, (_, i) => `mem_fake_${i}`),
      },
    });
    expect(tooManyIds.isError).toBe(true);

    await client.close();
  });
});
