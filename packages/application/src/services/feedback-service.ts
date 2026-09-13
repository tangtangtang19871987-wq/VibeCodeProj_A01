import type { DeliveryStage, FeedbackEventType, MemoryFeedback } from "@lams/domain";
import type { CreateFeedbackInput, FeedbackRepository } from "../ports/feedback.js";
import type { DeliveryRepository } from "../ports/recall.js";

export type RecordFeedbackInput = {
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  eventType: FeedbackEventType;
  note?: string;
  outcomeMetric?: Record<string, number>;
};

const STAGE_FOR_EVENT_TYPE: Record<FeedbackEventType, DeliveryStage> = {
  injected: "injected",
  useful: "used",
  not_useful: "used",
  harmful: "used",
  stale: "used",
  incorrect: "used",
  outcome_link: "outcome_linked",
};

/**
 * Records harness/human feedback (Section 12.2's memory_feedback tool).
 * Evidence level is never a caller-supplied parameter (ADR 0006): every
 * event recorded through this service is `reported`, because it always
 * arrives via an explicit call from outside the daemon. `inferred` is
 * reserved for the Milestone 5 evaluation runner, which writes through a
 * different path, not this one.
 */
export class FeedbackService {
  constructor(
    private readonly feedback: FeedbackRepository,
    private readonly delivery: DeliveryRepository,
  ) {}

  async record(input: RecordFeedbackInput): Promise<MemoryFeedback> {
    const feedbackInput: CreateFeedbackInput = { ...input, evidenceLevel: "reported" };
    const record = await this.feedback.create(feedbackInput);

    await this.delivery.record({
      sessionId: input.sessionId,
      recallTraceId: input.recallTraceId,
      memoryId: input.memoryId,
      stage: STAGE_FOR_EVENT_TYPE[input.eventType],
      evidenceLevel: "reported",
      note: input.note,
    });

    return record;
  }

  listForMemory(memoryId: string): Promise<MemoryFeedback[]> {
    return this.feedback.listForMemory(memoryId);
  }

  listForSession(sessionId: string): Promise<MemoryFeedback[]> {
    return this.feedback.listForSession(sessionId);
  }
}
