// Rehype plugin that rewrites bracketed citations before rehype-citation reads them.
// rehype-citation (citeproc, APA) has three gaps that this plugin closes in the source text:
//   1. A page range written with an en dash ("pp. 2–3") is not seen as plural and renders "p. 2–3".
//      The leading page range is passed with a hyphen; citeproc prints "pp. 2–3".
//   2. A locator that does not start with a page or a number ("Eq. 2.9, p. 9", "ch. 17") is
//      dropped or garbled. A page locator found later is moved to the front; otherwise the
//      locator is passed as a verbatim suffix ("@key, {}, ch. 17").
//   3. "Grinold and Easton [@grinold…]" renders the names twice. When the text just before the
//      bracket already gives the cited names, the author is suppressed: "Grinold and Easton (1998)".
// Usage in astro.config.mjs, before rehype-citation:
//   [rehypeCitationText, { bibliography: 'src/content/bibliography.bib' }]
import { readFileSync } from 'node:fs';
// The package exports its Cite class only through a folder mapping that Node no longer resolves.
const { default: Cite } = await import(new URL('cite.mjs', import.meta.resolve('rehype-citation')));

const SKIP = new Set(['code', 'pre', 'script', 'style']);
const BRACKET = /\[([^[\]]*@[^[\]]+)\]/g;
const ITEM = /^(?<prefix>[^@]*?)(?<dash>-?)@(?<key>[\p{L}\d_][^\s,{;]*[\p{L}\d_])(?<rest>.*)$/su;
const PAGE_LEAD = /^(?<label>pp?\.)\s*(?<from>\d+)\s*[–—-]\s*(?<to>\d+)/;
const PAGE_LATER = /,\s*(?<page>pp?\.\s*\d+(?:\s*[–—-]\s*\d+)?)(?=\s*,|\s*$)/;
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Family names that the in-text citation of each entry prints (authors, else editors). */
export function citedNames(bibliography) {
  const names = new Map();
  for (const item of new Cite(bibliography).data) {
    const people = item.author ?? item.editor ?? [];
    const families = people.map((p) => p.family ?? p.literal).filter(Boolean);
    if (families.length) names.set(item.id, families);
  }
  return names;
}

/** Whether `before` ends with the names of the cited entry, as a reader would write them. */
function endsWithNames(before, families) {
  const f = families.map(escape);
  const forms = f.length === 1 ? [f[0]]
    : f.length === 2 ? [`${f[0]} (?:and|&) ${f[1]}`, `${f[0]}[–-]${f[1]}`]
    : [`${f[0]} et al\\.?`, `${f.slice(0, -1).join(', ')},? and ${f.at(-1)}`, f.join('[–-]')];
  return new RegExp(`(?:^|[^\\p{L}])(?:${forms.join('|')})(?:['’]s)?\\s*$`, 'u').test(before);
}

/** Rewrite one locator so that rehype-citation keeps and labels it. */
export function normaliseLocator(locator) {
  let loc = locator.trim();
  if (!loc || loc.startsWith('{')) return loc;
  if (!/^(?:pp?\.\s*)?\d/.test(loc)) {
    const later = loc.match(PAGE_LATER);
    if (!later) return `{}, ${loc}`;
    loc = `${later.groups.page}, ${(loc.slice(0, later.index) + loc.slice(later.index + later[0].length)).trim()}`;
  }
  return loc.replace(PAGE_LEAD, (_, label, from, to) => `${label} ${from}-${to}`);
}

/** Rewrite the bracketed citations of one text, given the text that precedes it. */
export function normaliseCitations(text, names, preceding = '') {
  return text.replace(BRACKET, (whole, inner, offset) => {
    const parts = inner.split(';').map((part) => {
      const m = part.trim().match(ITEM);
      if (!m) return part.trim();
      const { prefix, dash, key, rest } = m.groups;
      const locator = rest.replace(/^\s*,/, '');
      const hasLocator = /^\s*,/.test(rest);
      return `${prefix}${dash}@${key}${hasLocator ? `, ${normaliseLocator(locator)}` : rest}`;
    });
    if (parts.length === 1) {
      const m = parts[0].match(ITEM);
      const families = m && names.get(m.groups.key);
      if (m && !m.groups.prefix && !m.groups.dash && families
        && endsWithNames(preceding + text.slice(0, offset), families)) parts[0] = `-${parts[0]}`;
    }
    return `[${parts.join('; ')}]`;
  });
}

export default function rehypeCitationText({ bibliography }) {
  const names = citedNames(readFileSync(bibliography, 'utf8'));
  const walk = (node) => {
    if (!node.children || SKIP.has(node.tagName)) return;
    let preceding = '';
    for (const child of node.children) {
      if (child.type === 'text') {
        if (child.value.includes('@')) child.value = normaliseCitations(child.value, names, preceding);
        preceding = child.value;
      } else {
        walk(child);
        preceding = '';
      }
    }
  };
  return (tree) => walk(tree);
}
