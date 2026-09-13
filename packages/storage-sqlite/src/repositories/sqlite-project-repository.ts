import type { ProjectRepository } from "@lams/application";
import { generateId, type Project } from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { ProjectRow } from "../schema.js";

function toDomain(row: ProjectRow): Project {
  return {
    id: row.id,
    key: row.key,
    name: row.name,
    description: row.description ?? undefined,
    createdAt: row.created_at,
  };
}

export class SqliteProjectRepository implements ProjectRepository {
  constructor(private readonly db: LamsDatabase) {}

  async create(input: {
    key: string;
    name: string;
    description?: string;
  }): Promise<Project> {
    const row: ProjectRow = {
      id: generateId("project"),
      key: input.key,
      name: input.name,
      description: input.description ?? null,
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("projects").values(row).execute();
    return toDomain(row);
  }

  async getById(id: string): Promise<Project | null> {
    const row = await this.db
      .selectFrom("projects")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirst();
    return row ? toDomain(row) : null;
  }

  async getByKey(key: string): Promise<Project | null> {
    const row = await this.db
      .selectFrom("projects")
      .selectAll()
      .where("key", "=", key)
      .executeTakeFirst();
    return row ? toDomain(row) : null;
  }

  async list(): Promise<Project[]> {
    const rows = await this.db
      .selectFrom("projects")
      .selectAll()
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(toDomain);
  }
}
