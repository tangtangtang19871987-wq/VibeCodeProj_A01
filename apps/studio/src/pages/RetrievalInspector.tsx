import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client.js";

const EVIDENCE_LABEL: Record<string, string> = {
  observed: "observed",
  reported: "reported",
  inferred: "inferred",
};

export function RetrievalInspector() {
  const { id } = useParams<{ id: string }>();
  const query = useQuery({
    queryKey: ["recall-candidates", id],
    queryFn: () => api.getRecallCandidates(id!),
    enabled: !!id,
  });

  if (query.isLoading) return <div className="panel">Loading…</div>;
  if (!query.data) return <div className="panel muted">Recall trace not found.</div>;

  const { trace, candidates, deliveryEvents } = query.data;
  const eligibleCount = candidates.filter((c) => c.eligible).length;
  const injectedCount = deliveryEvents.filter((e) => e.stage === "injected").length;
  const usedCount = deliveryEvents.filter((e) => e.stage === "used").length;

  return (
    <div className="panel-grid">
      <section className="panel">
        <h1>Retrieval Inspector</h1>
        <p className="memory-content">"{trace.query}"</p>
        <table className="kv-table">
          <tbody>
            <tr>
              <td>Session</td>
              <td>
                <Link to={`/sessions/${trace.sessionId}`}>{trace.sessionId}</Link>
              </td>
            </tr>
            <tr>
              <td>Strategy</td>
              <td>
                {trace.strategy} v{trace.strategyVersion}
              </td>
            </tr>
            <tr>
              <td>Status filter</td>
              <td>{trace.statusFilter.join(", ")}</td>
            </tr>
            <tr>
              <td>Budget</td>
              <td>
                {trace.budget.maxItems} items / {trace.budget.maxChars} chars — used{" "}
                {trace.returnedChars} chars (~{trace.estimatedTokens} tokens, estimated)
              </td>
            </tr>
            <tr>
              <td>Latency</td>
              <td>{trace.latencyMs.toFixed(2)} ms</td>
            </tr>
            <tr>
              <td>Created</td>
              <td>{new Date(trace.createdAt).toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Funnel</h2>
        <div className="funnel">
          <FunnelStage label="Candidates" value={trace.candidateCount} evidence="observed" />
          <FunnelStage label="Eligible" value={eligibleCount} evidence="observed" />
          <FunnelStage label="Selected" value={trace.selectedCount} evidence="observed" />
          <FunnelStage label="Returned" value={trace.returnedCount} evidence="observed" />
          <FunnelStage label="Reported injected" value={injectedCount} evidence="reported" unknown={injectedCount === 0} />
          <FunnelStage label="Reported used" value={usedCount} evidence="reported" unknown={usedCount === 0} />
        </div>
        <p className="muted">
          "Reported injected/used" comes only from explicit harness feedback
          (Milestone 3/5's memory_feedback tool). Zero here means{" "}
          <strong>unknown</strong>, not "not used" — this recall never claims
          more certainty than it has evidence for.
        </p>
      </section>

      <section className="panel">
        <h2>Candidates ({candidates.length})</h2>
        <div style={{ overflowX: "auto" }}>
          <table className="memory-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Memory</th>
                <th>Status</th>
                <th>Score</th>
                <th>Outcome</th>
                <th>Reasons</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr key={c.id}>
                  <td>{c.rank}</td>
                  <td>
                    <Link to={`/memories/${c.memoryId}`}>{c.title ?? c.memoryId}</Link>
                  </td>
                  <td className="muted">{c.statusAtQueryTime}</td>
                  <td className="muted">{c.finalScore.toFixed(3)}</td>
                  <td>
                    {c.selected ? (
                      <span className="status-badge status-badge--verified">selected</span>
                    ) : c.eligible ? (
                      <span className="status-badge status-badge--draft">eligible, not selected</span>
                    ) : (
                      <span className="status-badge status-badge--rejected">excluded</span>
                    )}
                  </td>
                  <td className="muted" style={{ fontSize: 11 }}>
                    {[...c.reasonCodes, ...c.exclusionReasons].join(", ") || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>Delivery evidence ({deliveryEvents.length})</h2>
        {deliveryEvents.length === 0 ? (
          <p className="muted">No delivery events recorded.</p>
        ) : (
          <ul className="plain-list">
            {deliveryEvents.map((e) => (
              <li key={e.id}>
                <strong>{e.stage}</strong>{" "}
                <span className={`evidence-tag evidence-tag--${e.evidenceLevel}`}>
                  {EVIDENCE_LABEL[e.evidenceLevel]}
                </span>{" "}
                <Link to={`/memories/${e.memoryId}`}>{e.memoryId}</Link>
                <div className="muted">{new Date(e.createdAt).toLocaleString()}</div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function FunnelStage({
  label,
  value,
  evidence,
  unknown,
}: {
  label: string;
  value: number;
  evidence: "observed" | "reported" | "inferred";
  unknown?: boolean;
}) {
  return (
    <div className="funnel-stage">
      <div className="funnel-stage__value">{unknown ? "?" : value}</div>
      <div className="funnel-stage__label">{label}</div>
      <span className={`evidence-tag evidence-tag--${evidence}`}>{evidence}</span>
    </div>
  );
}
