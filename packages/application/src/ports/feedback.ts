import type { EvidenceLevel, FeedbackEventType, MemoryFeedback } from "@lams/domain";

export type CreateFeedbackInput = {
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  eventType: FeedbackEventType;
  evidenceLevel: EvidenceLevel;
  note?: string;
  outcomeMetric?: Record<string, number>;
};

export interface FeedbackRepository {
  create(input: CreateFeedbackInput): Promise<MemoryFeedback>;
  listForMemory(memoryId: string): Promise<MemoryFeedback[]>;
  listForSession(sessionId: string): Promise<MemoryFeedback[]>;
}
