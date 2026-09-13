/**
 * Who created or acted on a record. Distinguishing human/agent/import/system
 * is required so the UI never implies agent output silently became
 * authoritative truth (PRD principle 4.2).
 */
export type ActorType = "human" | "agent" | "system" | "import";

export type ActorRef = {
  type: ActorType;
  id: string;
  displayName?: string;
};

export function humanActor(id: string, displayName?: string): ActorRef {
  return { type: "human", id, displayName };
}

export function agentActor(id: string, displayName?: string): ActorRef {
  return { type: "agent", id, displayName };
}

export function systemActor(id = "system"): ActorRef {
  return { type: "system", id };
}
