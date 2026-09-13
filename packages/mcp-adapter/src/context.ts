import type {
  FeedbackService,
  MemoryService,
  ProjectService,
  RetrievalService,
  SessionService,
} from "@lams/application";

/**
 * Everything the five MCP tools need, already composed by the daemon's
 * composition root (Section 8.2 — MCP is a thin adapter over the same
 * application services REST uses, never a second implementation of the
 * memory engine).
 */
export type McpToolContext = {
  projectService: ProjectService;
  sessionService: SessionService;
  memoryService: MemoryService;
  retrievalService: RetrievalService;
  feedbackService: FeedbackService;
};
