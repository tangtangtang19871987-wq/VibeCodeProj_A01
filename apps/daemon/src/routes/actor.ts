import { agentActor, humanActor, systemActor, type ActorRef } from "@lams/domain";
import { z } from "zod";

/**
 * v1 has no multi-user auth (Section 7.2, 18.2): REST callers may name an
 * actor explicitly (useful for seeding/demo data and for a future harness
 * calling REST directly), and default to a single local human actor
 * otherwise.
 */
export const actorInputSchema = z
  .object({
    actorType: z.enum(["human", "agent", "system", "import"]).optional(),
    actorId: z.string().min(1).optional(),
    actorDisplayName: z.string().optional(),
  })
  .optional();

export function resolveActor(input?: z.infer<typeof actorInputSchema>): ActorRef {
  if (!input?.actorType) return humanActor("local-user", "Local User");
  switch (input.actorType) {
    case "agent":
      return agentActor(input.actorId ?? "unknown-agent", input.actorDisplayName);
    case "system":
      return systemActor(input.actorId ?? "system");
    case "import":
      return { type: "import", id: input.actorId ?? "import", displayName: input.actorDisplayName };
    default:
      return humanActor(input.actorId ?? "local-user", input.actorDisplayName ?? "Local User");
  }
}
