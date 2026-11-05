// Slide model of the web slide deck (/<session>/slides/), without the DOM.
// A deck (src/content/slides/<session>.json) is a title slide, content slides in talk
// order and a closing slide. A content slide is read without the speaker (web/WRITING.md
// §6, "Slides"): its title states the claim; three to five bullets give the reason or
// the mechanism; it may add one formula, one key number with its label and one figure
// of the page; it ends on a takeaway line (shown after ▸) and a source line (the page
// section and the paper with its locator). It carries the speaker's notes and the page
// anchor it comes from. Bullets and the takeaway may hold inline math between $…$.
// The stage is a fixed 16:9 frame of 1920 × 1080 units scaled to the window.

import { wordCount } from './present-screens.js';

export const STAGE = { width: 1920, height: 1080 };
export const LIMITS = {
  titleWords: 14, minBullets: 3, bullets: 5, bulletWords: 15, takeawayWords: 14,
  minSlideWords: 60, slideWords: 80, coverWords: 30,
};

/** Words of a text, with each inline formula ($…$) counted as one word. */
export function textWords(text) {
  return wordCount(String(text ?? '').replace(/\$[^$]+\$/g, ' M '));
}

/**
 * Visible words of a slide: title, subtitle, author and date, bullets, number and
 * label, takeaway, and one word for a display formula. Figure labels and the source
 * line are not counted.
 */
export function slideWords(slide) {
  const parts = [slide.title, slide.subtitle, slide.author, slide.date, ...(slide.bullets ?? []), slide.takeaway];
  if (slide.number) parts.push(slide.number.value, slide.number.label);
  const formula = typeof slide.formula === 'string' && slide.formula.trim() ? 1 : 0;
  return parts.filter(Boolean).reduce((n, text) => n + textWords(text), formula);
}

// Writing rules of the slides profile (web/WRITING.md §6) that a pattern can check.
const STYLE = [
  [/—/, 'an em dash'],
  [/@[A-Za-z]|\[@/, 'a citation key'],
  [/\bnot\b[^.;]*,?\s*but\b/i, 'a "not X but Y" construction'],
  [/^(?:Select|Move|Click|Choose|Look|Note|Consider|See|Read|Compare|Use|Try|Open|Watch|Imagine|Let|Split|Buy|Sell|Shuffle|Remove|Drop|Check|Write)\b/, 'an imperative'],
];

// A paper named in a source line, "(2014" or "(1953;", must give a locator before ")".
const CITED_YEAR = /\((?:[^()]*?\b)?(?:1[89]|20)\d\d[a-z]?\b[^()]*\)/g;
const LOCATOR = /\b(?:pp?\.|Eqs?\.|Table|Exhibit|Figure|chap\.|Section|§)/;

/** Papers in a source line cited without a locator (page, equation, table…). */
export function citationsWithoutLocator(source) {
  return [...String(source ?? '').matchAll(CITED_YEAR)].map(m => m[0]).filter(c => !LOCATOR.test(c));
}

/**
 * Errors of a deck, as messages naming the slide (1-based) and the rule; empty when
 * the deck is valid. `anchors` are the page's ids, `figures` the ids of its embeds.
 * @param {{ slides: object[] }} deck
 * @param {{ anchors: Set<string>, figures: Set<string> }} page
 */
export function checkDeck(deck, { anchors, figures }) {
  const errors = [];
  const slides = deck?.slides;
  if (!Array.isArray(slides) || slides.length < 3) return ['the deck needs a title slide, content slides and a closing slide'];
  slides.forEach((slide, i) => {
    const at = `slide ${i + 1}`;
    const kind = slide.kind ?? 'content';
    const err = message => errors.push(`${at}: ${message}`);
    if (i === 0 && kind !== 'title') err('the first slide is the title slide');
    if (i === slides.length - 1 && kind !== 'closing') err('the last slide is the closing slide');
    if (i > 0 && i < slides.length - 1 && kind !== 'content') err(`a ${kind} slide inside the deck`);
    if (!['title', 'closing', 'content'].includes(kind)) err(`unknown kind "${kind}"`);
    if (typeof slide.title !== 'string' || !slide.title.trim()) err('no title');
    if (!Array.isArray(slide.notes) || !slide.notes.length || slide.notes.some(n => typeof n !== 'string' || !n.trim())) err('notes must be a non-empty list of lines');
    const visible = [slide.title, slide.subtitle, ...(Array.isArray(slide.bullets) ? slide.bullets : []), slide.number?.value, slide.number?.label, slide.takeaway, slide.source].filter(s => typeof s === 'string');
    for (const text of visible) for (const [pattern, name] of STYLE) if (pattern.test(text)) err(`${name} in "${text}"`);
    if (!anchors.has(slide.section)) err(`no page anchor "#${slide.section}"`);
    const words = slideWords(slide);
    if (kind !== 'content') {
      if (words > LIMITS.coverWords) err(`${words} words (at most ${LIMITS.coverWords} on a ${kind} slide)`);
      return;
    }
    if (words < LIMITS.minSlideWords || words > LIMITS.slideWords) err(`${words} words (${LIMITS.minSlideWords} to ${LIMITS.slideWords})`);
    if (textWords(slide.title ?? '') > LIMITS.titleWords) err(`title of ${textWords(slide.title)} words (at most ${LIMITS.titleWords})`);
    if (/\?\s*$/.test(slide.title ?? '')) err('the title is a question; it states the claim');
    if (!Array.isArray(slide.bullets) || slide.bullets.some(b => typeof b !== 'string' || !b.trim())) err('bullets must be a list of lines');
    else {
      if (slide.bullets.length < LIMITS.minBullets || slide.bullets.length > LIMITS.bullets) err(`${slide.bullets.length} bullets (${LIMITS.minBullets} to ${LIMITS.bullets})`);
      for (const b of slide.bullets) if (textWords(b) > LIMITS.bulletWords) err(`bullet of ${textWords(b)} words: "${b}"`);
    }
    if (typeof slide.takeaway !== 'string' || !slide.takeaway.trim()) err('no takeaway line');
    else if (textWords(slide.takeaway) > LIMITS.takeawayWords) err(`takeaway of ${textWords(slide.takeaway)} words (at most ${LIMITS.takeawayWords})`);
    if (typeof slide.source !== 'string' || !slide.source.trim()) err('no source line');
    else {
      if (!/§\d|Introduction|References|Code and data/.test(slide.source)) err(`the source line names no page section: "${slide.source}"`);
      for (const c of citationsWithoutLocator(slide.source)) err(`a paper without a locator in the source line: "${c}"`);
    }
    if (slide.formula !== undefined && (typeof slide.formula !== 'string' || !slide.formula.trim())) err('empty formula');
    if (slide.number !== undefined && !(typeof slide.number?.value === 'string' && typeof slide.number?.label === 'string')) err('number needs a value and a label');
    if (slide.figure !== undefined && !figures.has(slide.figure)) err(`no page figure "${slide.figure}"`);
  });
  return errors;
}

/**
 * A line of slide text as parts to render: plain text and inline formulas.
 * @param {string} text
 * @returns {{ math: boolean, text: string }[]}
 */
export function inlineParts(text) {
  return String(text).split(/(\$[^$]+\$)/).filter(Boolean).map(t => (/^\$[^$]+\$$/.test(t) ? { math: true, text: t.slice(1, -1) } : { math: false, text: t }));
}

/**
 * The page's figures, by id (the embed file name without .html), from the MDX source:
 * every <HtmlEmbed> and the section 1 figure (<AttributionFigure>, whose default panel
 * is the w-i1-attribution embed).
 * @param {string} mdx
 * @returns {Map<string, { src: string, data: string, title: string }>}
 */
export function pageFigures(mdx) {
  const attr = (tag, name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1];
  const figures = new Map();
  for (const [tag] of mdx.matchAll(/<HtmlEmbed\b[^>]*\/>/g)) {
    const src = attr(tag, 'src');
    figures.set(src.replace(/\.html$/, ''), { src, data: attr(tag, 'data'), title: attr(tag, 'title') ?? '' });
  }
  for (const [tag] of mdx.matchAll(/<AttributionFigure\b[^>]*\/>/g)) {
    figures.set('w-i1-attribution', { src: 'w-i1-attribution.html', data: 'w-i1-attribution.json', title: attr(tag, 'title') ?? '' });
  }
  return figures;
}

/** Ids written in the MDX source (id="…"), besides the heading ids. */
export function explicitIds(mdx) {
  return [...mdx.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
}

/** Label of the discreet link to the page: "§3.4" for a numbered heading, else the heading. */
export function sectionLabel(headingText) {
  const n = headingText.match(/^(\d+(?:\.\d+)*)\.?\s/);
  return n ? `§${n[1]}` : headingText;
}

/** Slide index (0-based) named by a URL hash "#n" (1-based); 0 when it names none. */
export function indexFromHash(hash, count) {
  const m = /^#?(\d+)$/.exec(hash ?? '');
  if (!m) return 0;
  return Math.max(0, Math.min(count - 1, Number(m[1]) - 1));
}

export const hashOf = index => `#${index + 1}`;

/** Clamped move from `index` by `delta` in a deck of `count` slides. */
export function stepTo(index, delta, count) {
  return Math.max(0, Math.min(count - 1, index + delta));
}

/**
 * What a key does: +1 or -1 (next, previous), 'first', 'last', or null.
 * Space goes forward, Shift+Space back.
 */
export function keyAction(key, shift = false) {
  if (['ArrowRight', 'ArrowDown', 'PageDown'].includes(key)) return 1;
  if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(key)) return -1;
  if (key === ' ') return shift ? -1 : 1;
  if (key === 'Home') return 'first';
  if (key === 'End') return 'last';
  return null;
}

/** A click at x on a stage of width `width`: -1 on the left edge, +1 on the right, else 0. */
export function edgeAction(x, width, edge = 0.12) {
  if (x < width * edge) return -1;
  if (x > width * (1 - edge)) return 1;
  return 0;
}

/** A finger swipe of (dx, dy): to the left goes forward, to the right back; else 0. */
export function swipeAction(dx, dy, min = 60) {
  if (Math.abs(dx) < min || Math.abs(dy) > Math.abs(dx) / 2) return 0;
  return dx < 0 ? 1 : -1;
}

/** Window pixels per stage unit: the 16:9 stage scaled to fit the window. */
export function stageUnit(width, height) {
  return Math.min(width / STAGE.width, height / STAGE.height);
}

/**
 * Where a slide's figure goes and its scale. The figure keeps its layout width `width`
 * and natural height `height`. `rooms` gives the room of each arrangement, in order of
 * preference: for example below the text, beside the text, or beside the title and
 * text at the full height of the slide. The figure takes the arrangement that shows it
 * largest; a later arrangement must show it `gain` times larger than an earlier one to
 * be chosen. The scale never exceeds `max`.
 * @param {{ width: number, height: number, rooms: Record<string, { width: number, height: number }>, max?: number, gain?: number }} p
 * @returns {{ layout: string, scale: number }}
 */
export function figureLayout({ width, height, rooms, max = 2.5, gain = 1.05 }) {
  const names = Object.keys(rooms);
  if (!(width > 0 && height > 0)) return { layout: names[0], scale: 1 };
  let best = null;
  for (const name of names) {
    const room = rooms[name];
    const scale = Math.max(0, Math.min(max, room.width / width, room.height / height));
    if (!best || scale > best.scale * gain) best = { layout: name, scale };
  }
  return best;
}

/**
 * The cue lines of a talk draft (talk/draft.md) without Markdown emphasis, for checking
 * that each note of a slide is taken from it.
 */
export function draftLines(draft) {
  return draft.split('\n').map(l => l.replace(/\*\*/g, '').replace(/^\s*-\s+/, '').trim()).filter(Boolean);
}

/** Notes of the deck that no line of the draft contains. */
export function notesMissingFromDraft(deck, draft) {
  const lines = draftLines(draft);
  const missing = [];
  deck.slides.forEach((slide, i) => {
    for (const note of slide.notes ?? []) if (!lines.some(l => l.includes(note))) missing.push(`slide ${i + 1}: "${note}"`);
  });
  return missing;
}
