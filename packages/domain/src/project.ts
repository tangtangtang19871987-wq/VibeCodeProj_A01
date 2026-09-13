export type Project = {
  id: string;
  key: string;
  name: string;
  description?: string;
  createdAt: string;
};

export type Agent = {
  id: string;
  name: string;
  harness: string;
  createdAt: string;
};

export type Artifact = {
  id: string;
  relativePath: string;
  contentHash: string;
  mimeType?: string;
  sizeBytes: number;
  createdAt: string;
};
