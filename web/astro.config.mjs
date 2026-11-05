import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeCitation from 'rehype-citation';
import rehypeCitationText from './tools/rehype-citation-text.mjs';
import { readFile, writeFile } from 'node:fs/promises';

// Opt-in for the local offline kit; public builds retain the pinned CDN URL.
const offlineRuntime = {
  name: 'local-pyodide-index',
  hooks: { 'astro:build:done': async ({ dir }) => {
    const index = process.env.PYODIDE_INDEX_URL;
    if (!index) return;
    if (!index.startsWith('/') || index.startsWith('//') || !index.endsWith('/'))
      throw new Error('PYODIDE_INDEX_URL must be an origin-relative directory');
    const worker = new URL('lab/worker.mjs', dir);
    const source = await readFile(worker, 'utf8');
    await writeFile(worker, source.replace(
      "'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/'", JSON.stringify(index)));
  } },
};
export default defineConfig({
  output: 'static',
  base: process.env.WEB_BASE || '/',
  integrations: [mdx(), offlineRuntime],
  devToolbar: { enabled: false },
  markdown: {
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' }, defaultColor: 'light', wrap: false },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeSlug, rehypeKatex,
      [rehypeCitationText, { bibliography: 'src/content/bibliography.bib' }],
      [rehypeCitation, { bibliography: 'src/content/bibliography.bib',
        linkCitations: true, csl: 'apa', suppressBibliography: true }]],
  },
});
