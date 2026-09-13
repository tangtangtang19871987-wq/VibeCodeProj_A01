import type {
  ActorRef,
  MemoryRelation,
  MemoryRelationType,
  Provenance,
  ProvenanceSourceType,
  ReviewAction,
  ReviewActionType,
} from "@lams/domain";

export type CreateProvenanceInput = {
  memoryId: string;
  memoryVersionId?: string;
  sourceType: ProvenanceSourceType;
  sourceRef: string;
  sessionId?: string;
  eventId?: string;
  excerpt?: string;
  artifactHash?: string;
};

export interface ProvenanceRepository {
  create(input: CreateProvenanceInput): Promise<Provenance>;
  listByMemoryId(memoryId: string): Promise<Provenance[]>;
}

export type CreateReviewActionInput = {
  memoryId: string;
  memoryVersionId: string;
  action: ReviewActionType;
  reason?: string;
  actor: ActorRef;
};

export interface ReviewRepository {
  create(input: CreateReviewActionInput): Promise<ReviewAction>;
  listByMemoryId(memoryId: string): Promise<ReviewAction[]>;
}

export type CreateMemoryRelationInput = {
  fromMemoryId: string;
  toMemoryId: string;
  type: MemoryRelationType;
  note?: string;
  createdBy: ActorRef;
};

export interface MemoryRelationRepository {
  create(input: CreateMemoryRelationInput): Promise<MemoryRelation>;
  listForMemory(memoryId: string): Promise<MemoryRelation[]>;
}
