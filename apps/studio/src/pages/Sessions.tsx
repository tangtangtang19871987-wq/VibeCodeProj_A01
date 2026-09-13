import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client.js";

export function Sessions() {
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: () => api.listSessions() });
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const queryClient = useQueryClient();

  const [projectId, setProjectId] = useState("");
  const [harness, setHarness] = useState("opencode");
  const [taskSummary, setTaskSummary] = useState("");

  const openSession = useMutation({
    mutationFn: api.openSession,
    onSuccess: () => {
      setTaskSummary("");
      void queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  return (
    <div className="panel-grid">
      <section className="panel">
        <h1>Sessions</h1>
        <p className="muted">
          Every recall, remembered candidate, and outcome is tracked against
          a session. Milestone 3's MCP tools will open sessions from real
          agent harnesses; until then, open one here to try recall manually.
        </p>
      </section>

      <section className="panel">
        <h2>Open a session</h2>
        <form
          className="stacked-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!projectId) return;
            openSession.mutate({ harness, projectId, taskSummary: taskSummary || undefined });
          }}
        >
          <div className="inline-form">
            <select aria-label="project" value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              <option value="">select a project…</option>
              {projects.data?.projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.key})
                </option>
              ))}
            </select>
            <input placeholder="harness (e.g. opencode)" value={harness} onChange={(e) => setHarness(e.target.value)} />
          </div>
          <input
            placeholder="task summary (optional)"
            value={taskSummary}
            onChange={(e) => setTaskSummary(e.target.value)}
          />
          <button type="submit" disabled={!projectId || openSession.isPending}>
            Open session
          </button>
        </form>
      </section>

      <section className="panel">
        {sessions.data?.sessions.length ? (
          <table className="memory-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Harness</th>
                <th>Status</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {sessions.data.sessions.map((s) => (
                <tr key={s.id}>
                  <td>
                    <Link to={`/sessions/${s.id}`}>{s.taskSummary ?? s.id}</Link>
                  </td>
                  <td className="muted">
                    {s.harness}
                    {s.agentName ? ` / ${s.agentName}` : ""}
                  </td>
                  <td>{s.status}</td>
                  <td className="muted">{new Date(s.startedAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No sessions yet.</p>
        )}
      </section>
    </div>
  );
}
