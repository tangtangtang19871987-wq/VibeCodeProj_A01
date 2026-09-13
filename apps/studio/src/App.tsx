import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./Layout.js";
import { Dashboard } from "./pages/Dashboard.js";
import { ComingSoon } from "./pages/ComingSoon.js";
import { MemoryExplorer } from "./pages/MemoryExplorer.js";
import { MemoryDetail } from "./pages/MemoryDetail.js";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/memories" element={<MemoryExplorer />} />
        <Route path="/memories/:id" element={<MemoryDetail />} />
        <Route
          path="/sessions"
          element={
            <ComingSoon
              title="Session Inspector"
              milestone="Milestone 2"
              explanation="Session timelines appear here once sessions and recall observability are implemented."
            />
          }
        />
        <Route
          path="/review-queue"
          element={
            <ComingSoon
              title="Review Queue"
              milestone="Milestone 1"
              explanation="Draft memories awaiting human review will appear here."
            />
          }
        />
        <Route
          path="/settings"
          element={
            <ComingSoon
              title="Settings & Diagnostics"
              milestone="Milestone 4"
              explanation="Data directory, backup/export, and retrieval strategy configuration will appear here."
            />
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
