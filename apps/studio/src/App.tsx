import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./Layout.js";
import { Dashboard } from "./pages/Dashboard.js";
import { ComingSoon } from "./pages/ComingSoon.js";
import { MemoryExplorer } from "./pages/MemoryExplorer.js";
import { MemoryDetail } from "./pages/MemoryDetail.js";
import { Sessions } from "./pages/Sessions.js";
import { SessionInspector } from "./pages/SessionInspector.js";
import { RetrievalInspector } from "./pages/RetrievalInspector.js";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/memories" element={<MemoryExplorer />} />
        <Route path="/memories/:id" element={<MemoryDetail />} />
        <Route path="/sessions" element={<Sessions />} />
        <Route path="/sessions/:id" element={<SessionInspector />} />
        <Route path="/recalls/:id" element={<RetrievalInspector />} />
        <Route
          path="/review-queue"
          element={
            <ComingSoon
              title="Review Queue"
              milestone="a future milestone"
              explanation="A dedicated one-by-one review queue is planned; for now, approve/reject/deprecate/supersede/merge all work from a memory's detail page in the Memory Explorer."
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
