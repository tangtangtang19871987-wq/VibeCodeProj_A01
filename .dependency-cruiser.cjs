/**
 * Architecture rules (Milestone 0 acceptance criteria): the domain layer
 * must not import Fastify, React, the MCP SDK, or SQLite details, and the
 * application layer must not import Fastify, React, the MCP SDK, or any
 * concrete storage package. Only the daemon/adapters are allowed to know
 * about all of them at once (Section 8.2).
 */
const FRAMEWORK_MODULES =
  "^(fastify|@fastify/.*|react|react-dom|react-router-dom|@modelcontextprotocol/sdk|better-sqlite3|kysely|vite|@vitejs/.*)$";

module.exports = {
  forbidden: [
    {
      name: "domain-no-framework-deps",
      comment:
        "packages/domain must stay pure TypeScript with no framework or storage dependency (PRD principle 4.9, Milestone 0 acceptance criteria).",
      severity: "error",
      from: { path: "^packages/domain" },
      to: { path: FRAMEWORK_MODULES, dependencyTypes: ["npm", "npm-dev"] },
    },
    {
      name: "domain-no-cross-package-deps",
      comment: "packages/domain must not depend on any other @lams/* package.",
      severity: "error",
      from: { path: "^packages/domain" },
      to: { path: "^packages/(?!domain)" },
    },
    {
      name: "application-no-framework-deps",
      comment:
        "packages/application defines use cases and ports only; it must not import Fastify/React/MCP/concrete storage (Section 8.2).",
      severity: "error",
      from: { path: "^packages/application" },
      to: { path: FRAMEWORK_MODULES, dependencyTypes: ["npm", "npm-dev"] },
    },
    {
      name: "application-no-storage-dep",
      comment:
        "packages/application must depend only on packages/domain (and packages/retrieval's pure functions), never on packages/storage-sqlite (dependency inversion: storage implements application's ports).",
      severity: "error",
      from: { path: "^packages/application" },
      to: { path: "^packages/storage-sqlite" },
    },
    {
      name: "retrieval-no-framework-or-storage-deps",
      comment:
        "packages/retrieval holds pure scoring/ranking functions (Section 11) with no framework, MCP, or storage dependency, so retrieval strategies stay unit-testable without a database.",
      severity: "error",
      from: { path: "^packages/retrieval" },
      to: { path: FRAMEWORK_MODULES, dependencyTypes: ["npm", "npm-dev"] },
    },
    {
      name: "retrieval-no-cross-package-deps",
      comment: "packages/retrieval may depend only on packages/domain.",
      severity: "error",
      from: { path: "^packages/retrieval" },
      to: { path: "^packages/(?!domain|retrieval)" },
    },
  ],
  options: {
    doNotFollow: { path: "node_modules" },
    tsPreCompilationDeps: true,
    tsConfig: { fileName: "tsconfig.base.json" },
    enhancedResolveOptions: { exportsFields: ["exports"], mainFields: ["main"] },
  },
};
