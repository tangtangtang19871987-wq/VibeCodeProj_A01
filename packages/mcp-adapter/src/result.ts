/**
 * Every tool returns a single JSON-serialized text content block (Section
 * 12.3: concise, bounded, no full debug data by default) plus, on error, a
 * stable machine-readable code the caller can branch on.
 */
export function textResult(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data) }] };
}

export function errorResult(code: string, message: string) {
  return {
    isError: true,
    content: [{ type: "text" as const, text: JSON.stringify({ error: { code, message } }) }],
  };
}
