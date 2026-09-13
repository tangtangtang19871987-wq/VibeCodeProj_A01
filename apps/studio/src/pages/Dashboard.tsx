import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client.js";

export function Dashboard() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const queryClient = useQueryClient();
  const [key, setKey] = useState("");
  const [name, setName] = useState("");

  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: () => {
      setKey("");
      setName("");
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  return (
    <div className="panel-grid">
      <section className="panel">
        <h1>Dashboard</h1>
        <p className="muted">
          What is happening in this memory system. Milestone 1 adds memory
          counts by kind/status, the review queue size, and recall
          activity — this view will grow as those land.
        </p>
      </section>

      <section className="panel">
        <h2>Daemon health</h2>
        {health.data ? (
          <table className="kv-table">
            <tbody>
              <tr>
                <td>Database</td>
                <td>{health.data.checks.database.ok ? "OK" : "DOWN"}</td>
              </tr>
              <tr>
                <td>Migrations</td>
                <td>
                  {health.data.checks.migrations.appliedMigrations.join(", ") ||
                    "none applied"}
                </td>
              </tr>
              <tr>
                <td>FTS5</td>
                <td>{health.data.checks.fts.ok ? "available" : "unavailable"}</td>
              </tr>
              <tr>
                <td>Artifact directory</td>
                <td>{health.data.checks.artifactDirectory.ok ? "writable" : "not writable"}</td>
              </tr>
            </tbody>
          </table>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </section>

      <section className="panel">
        <h2>Projects</h2>
        {projects.data?.projects.length ? (
          <ul className="plain-list">
            {projects.data.projects.map((p) => (
              <li key={p.id}>
                <strong>{p.name}</strong> <span className="muted">({p.key})</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">
            No projects yet. A project is created the first time a session or
            memory names it, or you can create one below.
          </p>
        )}

        <form
          className="inline-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!key || !name) return;
            createProject.mutate({ key, name });
          }}
        >
          <input
            placeholder="key (e.g. ilt-agent)"
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
          <input
            placeholder="display name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" disabled={createProject.isPending}>
            Create project
          </button>
        </form>
      </section>
    </div>
  );
}
