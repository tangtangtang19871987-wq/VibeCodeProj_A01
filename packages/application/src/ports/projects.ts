import type { Project } from "@lams/domain";

export interface ProjectRepository {
  create(input: { key: string; name: string; description?: string }): Promise<Project>;
  getById(id: string): Promise<Project | null>;
  getByKey(key: string): Promise<Project | null>;
  list(): Promise<Project[]>;
}
