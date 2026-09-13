import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client.js";

const EVENT_LABELS: Record<string, string> = {
  session_opened: "Session opened",
  session_updated: "Session updated",
  session_closed: "Session closed",
  recall_requested: "Recall requested",
  remember_proposed: "Memory proposed",
  feedback_received: "Feedback received",
  human_annotation: "Human annotation",
};

export function SessionInspector() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [includeDrafts, setIncludeDrafts] = useState(false);

  const detail = useQuery({
    queryKey: ["session", id],
    queryFn: () => api.getSession(id!),
    enabled: !!id,
  });

  const recall = useMutation({
    mutationFn: () => api.recall({ sessionId: id!, query, includeDrafts }),
    onSuccess: () => {
      setQuery("");
      void queryClient.invalidateQueries({ queryKey: ["session", id] });
    },
  });

  if (detail.isLoading) return <div className="panel">Loading…</div>;
  if (!detail.data) return <div className="panel muted">Session not found.</div>;

  const { session, events } = detail.data;

  return (
    <div className="panel-grid">
      <section className="panel">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h1>{session.taskSummary ?? session.id}</h1>
          <span className="status-badge">{session.status}</span>
        </div>
        <p className="muted">
          {session.harness}
          {session.agentName ? ` / ${session.agentName}` : ""} · project {session.projectId} ·
          started {new Date(session.startedAt).toLocaleString()}
        </p>
      </section>

      <section className="panel">
        <h2>Run a recall</h2>
        <p className="muted">
          Milestone 3's memory_recall MCP tool will do this from a real
          agent. This form exercises the same RetrievalService for manual
          testing.
        </p>
        <form
          className="inline-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!query) return;
            recall.mutate();
          }}
        >
          <input
            placeholder="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ flex: 1 }}
          />
          <label className="muted" style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={includeDrafts} onChange={(e) => setIncludeDrafts(e.target.checked)} />
            include drafts
          </label>
          <button type="submit" disabled={recall.isPending}>
            Recall
          </button>
        </form>
        {recall.data && (
          <p className="muted" style={{ marginTop: 8 }}>
            Returned {recall.data.packet.memories.length} memor
            {recall.data.packet.memories.length === 1 ? "y" : "ies"} —{" "}
            <Link to={`/recalls/${recall.data.trace.id}`}>inspect this recall</Link>
          </p>
        )}
      </section>

      <section className="panel">
        <h2>Timeline ({events.length} events)</h2>
        <p className="muted">
          Every event here is <strong>daemon-observed</strong> — recorded
          directly by LAMS, not reported by a harness. Harness-reported
          events (injected/used) arrive in Milestone 3/5.
        </p>
        <ul className="plain-list">
          {events.map((ev) => (
            <li key={ev.id}>
              <strong>{EVENT_LABELS[ev.type] ?? ev.type}</strong>{" "}
              <span className="evidence-tag evidence-tag--observed">observed</span>
              <div className="muted">{new Date(ev.createdAt).toLocaleString()}</div>
              {ev.type === "recall_requested" && typeof ev.payload.traceId === "string" && (
                <div>
                  <Link to={`/recalls/${ev.payload.traceId}`}>
                    "{String(ev.payload.query)}" — {String(ev.payload.returnedCount)}/
                    {String(ev.payload.candidateCount)} returned
                  </Link>
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
