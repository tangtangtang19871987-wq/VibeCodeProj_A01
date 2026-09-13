import type { Project } from "@lams/domain";
import type { ProjectRepository } from "../ports/projects.js";

export class ProjectService {
  constructor(private readonly projects: ProjectRepository) {}

  async create(input: { key: string; name: string; description?: string }): Promise<Project> {
    const existing = await this.projects.getByKey(input.key);
    if (existing) return existing;
    return this.projects.create(input);
  }

  getById(id: string): Promise<Project | null> {
    return this.projects.getById(id);
  }

  list(): Promise<Project[]> {
    return this.projects.list();
  }
}
