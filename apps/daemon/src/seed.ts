/**
 * Demo seed data matching the PRD Section 27 demonstration scenario: an
 * ilt-agent project with a superseded "always use Adam" lesson, the
 * verified lesson that replaced it, a draft awaiting review, and a
 * rejected fact — enough to exercise every status and the Explorer's
 * filters without a real agent session.
 *
 * Run with: pnpm --filter @lams/daemon run seed
 */
import { humanActor, agentActor } from "@lams/domain";
import { loadConfig } from "./config.js";
import { createContext } from "./context.js";

async function main() {
  const config = loadConfig();
  const ctx = createContext(config);
  const human = humanActor("local-user", "Local User");
  const agent = agentActor("opencode-demo", "OpenCode (demo)");

  try {
    const project = await ctx.projectService.create({
      key: "ilt-agent",
      name: "ILT Agent",
      description: "Demo project used in the PRD's Section 27 walkthrough.",
    });

    const oldLesson = await ctx.memoryService.create({
      kind: "lesson",
      scope: { level: "project", id: project.key },
      status: "verified",
      createdBy: human,
      version: {
        title: "Always use Adam",
        content: "Always use the Adam optimizer for contact-mechanics cases.",
        tags: ["optimizer", "contact-mechanics"],
      },
    });

    const newLesson = await ctx.memoryService.create({
      kind: "lesson",
      scope: { level: "project", id: project.key },
      status: "draft",
      createdBy: agent,
      version: {
        title: "Adam warm start then LBFGS",
        content:
          "For dense-contact cases with early oscillation, use a short Adam warm start and then switch to LBFGS.",
        tags: ["optimizer", "contact-mechanics", "lbfgs"],
      },
    });
    await ctx.memoryService.addProvenance({
      memoryId: newLesson.memory.id,
      sourceType: "session_event",
      sourceRef: "demo-session-A",
      excerpt: "Adam diverged after ~200 steps; switching to LBFGS converged in 40 more.",
    });
    await ctx.memoryService.approve(newLesson.memory.id, human, "Edited condition to avoid overgeneralizing; approved for project scope.");
    await ctx.memoryService.supersede(
      oldLesson.memory.id,
      newLesson.memory.id,
      human,
      "Superseded by more specific guidance for dense-contact cases.",
    );

    const draftWarning = await ctx.memoryService.create({
      kind: "warning",
      scope: { level: "project", id: project.key },
      createdBy: agent,
      version: {
        title: "Mesh refinement can silently diverge",
        content:
          "Refining the contact mesh past 0.5um without also tightening the solver tolerance can silently diverge instead of erroring.",
        tags: ["mesh", "solver"],
      },
    });
    void draftWarning;

    const rejectedFact = await ctx.memoryService.create({
      kind: "fact",
      scope: { level: "project", id: project.key },
      createdBy: agent,
      version: {
        title: "Incorrect: max iterations is 50",
        content: "The solver's default max iteration count is 50.",
      },
    });
    await ctx.memoryService.reject(
      rejectedFact.memory.id,
      human,
      "Incorrect — the default is 500, not 50. Agent misread the config comment.",
    );

    console.log("Seed complete:");
    console.log(`  project: ${project.key}`);
    console.log(`  superseded lesson: ${oldLesson.memory.id}`);
    console.log(`  verified lesson:   ${newLesson.memory.id}`);
    console.log(`  draft warning:     ${draftWarning.memory.id}`);
    console.log(`  rejected fact:     ${rejectedFact.memory.id}`);
  } finally {
    ctx.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
