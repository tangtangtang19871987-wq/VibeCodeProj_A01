import type {
  DeliveryEvent,
  DeliveryStage,
  EvidenceLevel,
  RecallCandidate,
  RecallTrace,
} from "@lams/domain";

export interface RecallRepository {
  createTraceWithCandidates(
    trace: RecallTrace,
    candidates: RecallCandidate[],
  ): Promise<void>;
  getTrace(id: string): Promise<RecallTrace | null>;
  listCandidates(traceId: string): Promise<RecallCandidate[]>;
  listTracesForSession(sessionId: string): Promise<RecallTrace[]>;
}

export type RecordDeliveryEventInput = {
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  memoryVersionId?: string;
  stage: DeliveryStage;
  evidenceLevel: EvidenceLevel;
  note?: string;
};

export interface DeliveryRepository {
  record(input: RecordDeliveryEventInput): Promise<DeliveryEvent>;
  listForSession(sessionId: string): Promise<DeliveryEvent[]>;
  listForTrace(recallTraceId: string): Promise<DeliveryEvent[]>;
}
