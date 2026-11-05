// Screen model of the presentation view, without the DOM (web/WRITING.md §6).
// One screen per h2 opening and per h3 subsection, in page order. A screen shows its
// heading, one message sentence and its first figure; the rest of its content is
// folded below. The talk may visit the screens in another order (talk/draft.md).

/**
 * @typedef {{ tag: string, id?: string, text?: string, figure?: boolean }} Block  a top-level
 *   element of the page; `figure` when it is or holds a figure outside a fold
 * @typedef {{ slug: string, title: string, level: number, heading: number | null,
 *   figure: number | null, blocks: number[] }} Screen  indices refer to the block list
 */

const HEADINGS = { H2: 2, H3: 3 };

/**
 * Splits the page's top-level blocks into screens. Content before the first heading
 * gets a screen of its own, titled `introTitle`. A screen's figure is its first
 * block that is or holds a figure; `blocks` lists every block of the screen, heading and figure
 * included, in page order.
 * @param {Block[]} blocks
 * @param {string} introTitle
 * @returns {Screen[]}
 */
export function screensOf(blocks, introTitle = '') {
  /** @type {Screen[]} */
  const screens = [];
  let current = null;
  blocks.forEach((block, i) => {
    const level = HEADINGS[block.tag];
    if (level) {
      current = { slug: block.id ?? '', title: (block.text ?? '').trim(), level, heading: i, figure: null, blocks: [] };
      screens.push(current);
    } else if (!current) {
      current = { slug: 'intro', title: introTitle, level: 2, heading: null, figure: null, blocks: [] };
      screens.push(current);
    }
    current.blocks.push(i);
    if ((block.figure || block.tag === 'FIGURE') && current.figure === null) current.figure = i;
  });
  return screens;
}

/** Heading text as written in the message list: straight quotes, single spaces. */
export function normaliseHeading(text) {
  return text.replace(/&amp;/g, '&').replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, ' ').trim();
}

/**
 * The headings that need a message: every heading up to the last numbered one
 * (the introduction and sections 1 to 5). The unnumbered headings after it (code,
 * references, reuse, citation) are back matter: their screens carry no message.
 * @param {{ text: string }[]} headings in page order
 */
export function bodyHeadings(headings) {
  let last = -1;
  headings.forEach((h, i) => { if (/^\d/.test(normaliseHeading(h.text))) last = i; });
  return headings.slice(0, last + 1);
}

/**
 * Maps the message list (keyed by heading text) to heading ids. Throws, naming each
 * one, if a body heading has no message or a message names no heading.
 * @param {{ id: string, text: string }[]} headings in page order
 * @param {Record<string, string>} messages heading text → message; keys starting with "_" are comments
 * @returns {Record<string, string>} heading id → message
 */
export function messagesById(headings, messages) {
  const byText = new Map(headings.map(h => [normaliseHeading(h.text), h.id]));
  const keys = Object.keys(messages).filter(k => !k.startsWith('_'));
  const known = new Set(keys.map(normaliseHeading));
  const errors = [];
  for (const h of bodyHeadings(headings)) {
    if (!known.has(normaliseHeading(h.text))) errors.push(`heading without a message: "${h.text}"`);
  }
  /** @type {Record<string, string>} */
  const out = {};
  for (const key of keys) {
    const id = byText.get(normaliseHeading(key));
    if (id === undefined) errors.push(`message without a heading: "${key}"`);
    else out[id] = messages[key];
  }
  if (errors.length) throw new Error(`Presentation messages do not match the headings:\n  ${errors.join('\n  ')}`);
  return out;
}

/**
 * The talk's screens, in the order the talk visits them: the title screen, the
 * anchors linked by every "**Page:**" paragraph of the talk draft (it runs to the
 * next blank line), in order, then the title screen again (the closing screen with
 * the QR code). The title screen is "".
 * @param {string} draft Markdown of talk/draft.md
 * @returns {string[]} anchor ids
 */
export function talkOrder(draft) {
  const anchors = [];
  for (const paragraph of draft.split(/\n\s*\n/)) {
    if (!/^\*\*Page:\*\*/.test(paragraph.trim())) continue;
    for (const m of paragraph.matchAll(/\]\([^)#\s]*#([^)\s]+)\)/g)) anchors.push(decodeURIComponent(m[1]));
  }
  if (!anchors.length) throw new Error('The talk draft has no "**Page:**" line with a link');
  return ['', ...anchors, ''];
}

/**
 * Screen indices to visit, given the screen holding each anchor (`screenOf` returns
 * -1 for an unknown anchor, which is an error). Page order is every screen in turn.
 * @param {number} count number of screens, title screen included
 * @param {string[] | null} anchors talk order, or null for page order
 * @param {(anchor: string) => number} screenOf
 */
export function visitOrder(count, anchors, screenOf) {
  if (!anchors) return Array.from({ length: count }, (_, i) => i);
  return anchors.map(a => {
    const i = screenOf(a);
    if (i < 0) throw new Error(`Talk order: no screen holds "#${a}"`);
    return i;
  });
}

/**
 * Where the order stands once `screen` is shown, coming from position `from`: the
 * first position at or after `from` that shows it, else the first anywhere, else
 * null (a screen the order skips: a detour, reached by a link).
 * @param {number[]} order
 */
export function positionOf(order, screen, from = 0) {
  const ahead = order.indexOf(screen, Math.max(0, from));
  if (ahead >= 0) return ahead;
  const any = order.indexOf(screen);
  return any >= 0 ? any : null;
}

/** Words in a text: whitespace-separated tokens holding a letter or a digit. */
export function wordCount(text) {
  return text.split(/\s+/).filter(t => /[\p{L}\p{N}]/u.test(t)).length;
}

/**
 * The derived data file of the presentation view (src/content/present.json): the
 * messages without comment keys, and the talk order.
 * @param {Record<string, string>} messages talk/present-messages.json
 * @param {string} draft talk/draft.md
 */
export function presentData(messages, draft) {
  return {
    _comment: 'Derived by tools/present-data.mjs from the talk messages and the talk draft. Do not edit.',
    messages: Object.fromEntries(Object.entries(messages).filter(([k]) => !k.startsWith('_'))),
    talkOrder: talkOrder(draft),
  };
}

/**
 * Scale and layout of a screen's figure. The figure keeps its layout width `width`
 * and natural height `height`. Stacked below the message it has `stackRoom` of
 * height; beside the heading and message it has `sideRoom` of height and leaves a
 * text column of at least `textMin` (with `gap`). Beside wins when it shows the
 * figure at least `gain` times larger.
 * @returns {{ side: boolean, scale: number }}
 */
export function figureFit({ width, height, stackRoom, sideRoom, textMin = 420, gap = 40, gain = 1.2 }) {
  if (!(height > 0 && width > 0)) return { side: false, scale: 1 };
  const stacked = Math.min(1, stackRoom / height);
  const beside = Math.min(1, sideRoom / height, (width - textMin - gap) / width);
  return beside > stacked * gain ? { side: true, scale: beside } : { side: false, scale: stacked };
}
