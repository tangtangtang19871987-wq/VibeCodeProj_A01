import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client.js";
import { StatusBadge } from "./MemoryExplorer.js";

export function MemoryDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [reason, setReason] = useState("");
  const [targetMemoryId, setTargetMemoryId] = useState("");
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");

  const detail = useQuery({
    queryKey: ["memory", id],
    queryFn: () => api.getMemory(id!),
    enabled: !!id,
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["memory", id] });
    void queryClient.invalidateQueries({ queryKey: ["memories"] });
  };

  const review = useMutation({
    mutationFn: (input: { action: string; reason?: string; targetMemoryId?: string }) =>
      api.reviewAction(id!, input),
    onSuccess: invalidate,
  });

  const addVersion = useMutation({
    mutationFn: (input: { title: string; content: string; changeReason?: string }) =>
      api.addVersion(id!, input),
    onSuccess: () => {
      setEditing(false);
      invalidate();
    },
  });

  const del = useMutation({
    mutationFn: () => api.deleteMemory(id!),
    onSuccess: () => {
      invalidate();
      navigate("/memories");
    },
  });

  if (detail.isLoading) return <div className="panel">Loading…</div>;
  if (detail.error || !detail.data) {
    return (
      <div className="panel">
        <p className="muted">Memory not found (it may have been permanently removed).</p>
        <Link to="/memories">Back to Memory Explorer</Link>
      </div>
    );
  }

  const { memory, currentVersion, versions, provenance, reviews, relations } = detail.data;
  const canApprove = memory.status === "draft";
  const canReject = memory.status === "draft" || memory.status === "verified";
  const canDeprecate = memory.status === "draft" || memory.status === "verified";
  const canRestore = memory.status === "rejected" || memory.status === "deprecated";
  const canSupersedeOrMerge = memory.status === "draft" || memory.status === "verified";

  return (
    <div className="panel-grid">
      <section className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1>{currentVersion.title}</h1>
          <StatusBadge status={memory.status} />
        </div>
        <p className="muted">
          {memory.kind} · {memory.scopeDisplay} · v{currentVersion.version} · updated{" "}
          {new Date(memory.updatedAt).toLocaleString()}
        </p>
        {memory.supersededById && (
          <p className="muted">
            Superseded by{" "}
            <Link to={`/memories/${memory.supersededById}`}>{memory.supersededById}</Link>
          </p>
        )}
        {memory.mergedIntoId && (
          <p className="muted">
            Merged into <Link to={`/memories/${memory.mergedIntoId}`}>{memory.mergedIntoId}</Link>
          </p>
        )}

        {editing ? (
          <form
            className="stacked-form"
            onSubmit={(e) => {
              e.preventDefault();
              addVersion.mutate({ title: editTitle, content: editContent, changeReason: reason });
            }}
          >
            <input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
            <textarea value={editContent} onChange={(e) => setEditContent(e.target.value)} rows={4} />
            <input placeholder="change reason" value={reason} onChange={(e) => setReason(e.target.value)} />
            <div className="inline-form">
              <button type="submit" disabled={addVersion.isPending}>
                Save as new version
              </button>
              <button type="button" onClick={() => setEditing(false)}>
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <>
            <p className="memory-content">{currentVersion.content}</p>
            {currentVersion.tags.length > 0 && (
              <p className="muted">tags: {currentVersion.tags.join(", ")}</p>
            )}
          </>
        )}
      </section>

      <section className="panel">
        <h2>Review actions</h2>
        <p className="muted">
          Every action here is recorded in the audit trail below and, where
          applicable, transitions the memory's status (Section 9.1/15.2).
        </p>
        <div className="inline-form" style={{ flexWrap: "wrap", marginBottom: 8 }}>
          <input
            placeholder="reason (optional)"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            style={{ flex: 1, minWidth: 200 }}
          />
        </div>
        <div className="inline-form" style={{ flexWrap: "wrap" }}>
          {!editing && (
            <button
              onClick={() => {
                setEditTitle(currentVersion.title);
                setEditContent(currentVersion.content);
                setEditing(true);
              }}
            >
              Edit (new version)
            </button>
          )}
          {canApprove && (
            <button onClick={() => review.mutate({ action: "approve", reason })}>Approve</button>
          )}
          {canReject && (
            <button onClick={() => review.mutate({ action: "reject", reason })}>Reject</button>
          )}
          {canDeprecate && (
            <button onClick={() => review.mutate({ action: "deprecate", reason })}>Deprecate</button>
          )}
          {canRestore && (
            <button onClick={() => review.mutate({ action: "restore", reason })}>Restore to draft</button>
          )}
          {memory.status !== "deleted" && (
            <button className="danger" onClick={() => del.mutate()}>
              Delete (tombstone)
            </button>
          )}
        </div>

        {canSupersedeOrMerge && (
          <div className="inline-form" style={{ marginTop: 12, flexWrap: "wrap" }}>
            <input
              placeholder="other memory id (supersede/merge target)"
              value={targetMemoryId}
              onChange={(e) => setTargetMemoryId(e.target.value)}
              style={{ flex: 1, minWidth: 260 }}
            />
            <button
              disabled={!targetMemoryId}
              onClick={() => review.mutate({ action: "supersede", targetMemoryId, reason })}
            >
              This memory is superseded by →
            </button>
            <button
              disabled={!targetMemoryId}
              onClick={() => review.mutate({ action: "merge", targetMemoryId, reason })}
            >
              Merge this memory into →
            </button>
          </div>
        )}
        {review.isError && <p className="error-text">{(review.error as Error).message}</p>}
      </section>

      <section className="panel">
        <h2>Version history ({versions.length})</h2>
        <ul className="plain-list">
          {[...versions].reverse().map((v) => (
            <li key={v.id}>
              <strong>v{v.version}</strong> — {v.title}
              <div className="muted">
                {new Date(v.createdAt).toLocaleString()} by {v.createdBy.displayName ?? v.createdBy.id}
                {v.changeReason ? ` — ${v.changeReason}` : ""}
              </div>
              {v.id !== currentVersion.id && <div className="muted">{v.content}</div>}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>Provenance ({provenance.length})</h2>
        {provenance.length === 0 ? (
          <p className="muted">No evidence linked yet.</p>
        ) : (
          <ul className="plain-list">
            {provenance.map((p) => (
              <li key={p.id}>
                <strong>{p.sourceType}</strong>: {p.sourceRef}
                {p.excerpt && <div className="muted">"{p.excerpt}"</div>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Review history ({reviews.length})</h2>
        {reviews.length === 0 ? (
          <p className="muted">No review actions yet.</p>
        ) : (
          <ul className="plain-list">
            {reviews.map((r) => (
              <li key={r.id}>
                <strong>{r.action}</strong> by {r.actor.displayName ?? r.actor.id} (
                {r.actor.type}) — {new Date(r.createdAt).toLocaleString()}
                {r.reason && <div className="muted">{r.reason}</div>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Lineage ({relations.length})</h2>
        {relations.length === 0 ? (
          <p className="muted">No relations to other memories yet.</p>
        ) : (
          <ul className="plain-list">
            {relations.map((r) => (
              <li key={r.id}>
                <Link to={`/memories/${r.fromMemoryId}`}>{r.fromMemoryId}</Link> —{" "}
                <strong>{r.type}</strong> →{" "}
                <Link to={`/memories/${r.toMemoryId}`}>{r.toMemoryId}</Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
