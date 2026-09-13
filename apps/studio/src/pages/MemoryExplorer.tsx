import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type MemoryKind, type MemoryStatus } from "../api/client.js";

const ALL_KINDS: MemoryKind[] = ["fact", "decision", "preference", "episode", "warning", "lesson"];
const ALL_STATUSES: MemoryStatus[] = ["draft", "verified", "rejected", "superseded", "deprecated", "deleted"];

export function MemoryExplorer() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const statusFilter = (params.get("status")?.split(",").filter(Boolean) ?? []) as MemoryStatus[];
  const kindFilter = (params.get("kind")?.split(",").filter(Boolean) ?? []) as MemoryKind[];
  const includeDeleted = params.get("includeDeleted") === "true";

  const query = useQuery({
    queryKey: ["memories", q, statusFilter, kindFilter, includeDeleted],
    queryFn: () =>
      api.listMemories({
        q: q || undefined,
        status: statusFilter.length ? statusFilter : undefined,
        kind: kindFilter.length ? kindFilter : undefined,
        includeDeleted,
      }),
  });

  function updateParam(key: string, value: string | null) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }

  function toggleMulti(key: string, current: string[], value: string) {
    const next = current.includes(value) ? current.filter((v) => v !== value) : [...current, value];
    updateParam(key, next.length ? next.join(",") : null);
  }

  return (
    <div className="panel-grid">
      <section className="panel">
        <h1>Memory Explorer</h1>
        <input
          className="search-box"
          placeholder="Full-text search…"
          defaultValue={q}
          onChange={(e) => updateParam("q", e.target.value || null)}
        />

        <div className="filter-row">
          <div>
            <span className="muted">Status: </span>
            {ALL_STATUSES.map((s) => (
              <button
                key={s}
                className={"chip" + (statusFilter.includes(s) ? " chip--active" : "")}
                onClick={() => toggleMulti("status", statusFilter, s)}
              >
                {s}
              </button>
            ))}
          </div>
          <div>
            <span className="muted">Kind: </span>
            {ALL_KINDS.map((k) => (
              <button
                key={k}
                className={"chip" + (kindFilter.includes(k) ? " chip--active" : "")}
                onClick={() => toggleMulti("kind", kindFilter, k)}
              >
                {k}
              </button>
            ))}
          </div>
          <label className="muted" style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={includeDeleted}
              onChange={(e) => updateParam("includeDeleted", e.target.checked ? "true" : null)}
            />
            Show deleted
          </label>
        </div>

        {statusFilter.length === 0 && (
          <p className="muted">
            {q
              ? "Full-text search defaults to draft + verified memories. Use the status filters to also search rejected, superseded, or deprecated memories."
              : "Browsing shows every non-deleted memory, including drafts, rejected, and superseded ones. Use the status filters to narrow."}
          </p>
        )}
      </section>

      <section className="panel">
        {query.isLoading && <p className="muted">Loading…</p>}
        {query.data && query.data.items.length === 0 && (
          <p className="muted">
            No memories match. Memories appear here once created via the API,
            MCP (Milestone 3), or the form below.
          </p>
        )}
        {query.data && query.data.items.length > 0 && (
          <table className="memory-table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Kind</th>
                <th>Status</th>
                <th>Scope</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map(({ memory, currentVersion }) => (
                <tr key={memory.id}>
                  <td>
                    <Link to={`/memories/${memory.id}`}>{currentVersion.title}</Link>
                  </td>
                  <td>{memory.kind}</td>
                  <td>
                    <StatusBadge status={memory.status} />
                  </td>
                  <td className="muted">{memory.scopeDisplay}</td>
                  <td className="muted">{new Date(memory.updatedAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <CreateMemoryPanel />
    </div>
  );
}

export function StatusBadge({ status }: { status: MemoryStatus }) {
  return <span className={`status-badge status-badge--${status}`}>{status}</span>;
}

function CreateMemoryPanel() {
  const queryClient = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const [kind, setKind] = useState<MemoryKind>("fact");
  const [scopeLevel, setScopeLevel] = useState("project");
  const [scopeId, setScopeId] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  const create = useMutation({
    mutationFn: api.createMemory,
    onSuccess: () => {
      setTitle("");
      setContent("");
      void queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  return (
    <section className="panel">
      <h2>Create a memory</h2>
      <p className="muted">
        Created here as a human-authored memory (draft by default). Milestone 3
        adds agent-created memories via MCP's memory_remember tool.
      </p>
      <form
        className="stacked-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (!title || !content || !scopeId) return;
          create.mutate({
            kind,
            scope: { level: scopeLevel, id: scopeId },
            version: { title, content },
          });
        }}
      >
        <div className="inline-form">
          <select
            aria-label="kind"
            value={kind}
            onChange={(e) => setKind(e.target.value as MemoryKind)}
          >
            {ALL_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <select
            aria-label="scope level"
            value={scopeLevel}
            onChange={(e) => {
              setScopeLevel(e.target.value);
              setScopeId("");
            }}
          >
            <option value="project">project</option>
            <option value="agent">agent</option>
            <option value="session">session</option>
          </select>
          {scopeLevel === "project" ? (
            <select aria-label="scope project" value={scopeId} onChange={(e) => setScopeId(e.target.value)}>
              <option value="">select a project…</option>
              {projects.data?.projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.key})
                </option>
              ))}
            </select>
          ) : (
            <input
              placeholder={`${scopeLevel} id`}
              value={scopeId}
              onChange={(e) => setScopeId(e.target.value)}
            />
          )}
        </div>
        <input placeholder="title" value={title} onChange={(e) => setTitle(e.target.value)} />
        <textarea placeholder="content" value={content} onChange={(e) => setContent(e.target.value)} rows={3} />
        <button type="submit" disabled={create.isPending}>
          Create draft memory
        </button>
      </form>
    </section>
  );
}
