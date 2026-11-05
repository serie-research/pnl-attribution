// The papers of the post, from src/content/assets/data/papers.json.
import data from "../../content/assets/data/papers.json";

export type Appearance = { section: string; as: "note" | "cite" };
export type Paper = {
  key: string;
  authors: string;
  year: number;
  title: string;
  venue: string;
  note?: string;
  url?: string;
  pdf?: string;
  appears: Appearance[];
};

export const papers = data.papers as Paper[];

export function paper(key: string): Paper {
  const found = papers.find((p) => p.key === key);
  if (!found) throw new Error(`No paper "${key}" in papers.json`);
  return found;
}
