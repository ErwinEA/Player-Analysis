/** Split formatted insight text into display blocks (one per bullet / caveat). */
export function parseInsightBlocks(summary: string): string[] {
  return summary
    .replace(/\r\n/g, "\n")
    .split(/\n\n+/)
    .map((block) => block.trim())
    .filter(Boolean);
}
