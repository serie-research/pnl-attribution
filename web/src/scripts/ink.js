// Ink model of the presentation view: strokes per section, without the DOM.
// A stroke's points are flat [x0, y0, w0, x1, y1, w1, ...]: content coordinates of
// the section and the line width at that point. Each section keeps a history of
// actions so that undo reverts strokes, erasures and clears alike.

/** @typedef {'pen' | 'highlighter'} Tool */
/** @typedef {{ tool: Tool, colour: string, points: number[] }} Stroke */
/** @typedef {{ pointerType: string, pressure: number, x: number, y: number }} Sample */

const MOUSE_WIDTH = 2.5;
const HIGHLIGHTER_WIDTH = 18;

/** A stylus always draws; a mouse when ink is armed; a finger never, so that it scrolls. */
export function draws(pointerType, armed) {
  return pointerType === 'pen' || (pointerType === 'mouse' && armed);
}

/** PointerEvent.buttons bit of a pen's eraser end (Surface, Wacom; the Apple Pencil has none). */
export const ERASER_BUTTON = 32;

/**
 * Whether a pointer erases rather than writes: with the eraser tool selected, or a pen
 * held by its eraser end, whatever tool is selected.
 */
export function erases(tool, pointerType, buttons) {
  return tool === 'eraser' || (pointerType === 'pen' && (buttons & ERASER_BUTTON) !== 0);
}

/**
 * Whether the page holds a touch back from scrolling and selecting, given the touch
 * types of the changed touches (Safari's Touch.touchType: 'stylus' or 'direct') and
 * whether a stroke is under way: the pencil always, any touch (a resting palm) while
 * writing, since a scroll mid-stroke would move the content under the pencil.
 */
export function holdsTouch(touchTypes, writing) {
  return writing || touchTypes.includes('stylus');
}

/** Line width at one point. */
export function width(tool, pointerType, pressure) {
  if (tool === 'highlighter') return HIGHLIGHTER_WIDTH;
  if (pointerType !== 'pen') return MOUSE_WIDTH;
  return 0.8 + 3.6 * (pressure > 0 ? pressure : 0.5);
}

export function createInk() {
  return { sections: new Map() };
}

function section(ink, key) {
  if (!ink.sections.has(key)) ink.sections.set(key, { strokes: [], history: [] });
  return ink.sections.get(key);
}

/** Starts a stroke; returns null for a finger. */
export function begin(ink, key, sample, { tool, colour }) {
  if (sample.pointerType === 'touch') return null;
  /** @type {Stroke} */
  const stroke = { tool, colour, pointerType: sample.pointerType, points: [] };
  extend(stroke, [sample]);
  const s = section(ink, key);
  s.strokes.push(stroke);
  s.history.push({ type: 'add', stroke });
  return stroke;
}

export function extend(stroke, samples) {
  for (const { x, y, pressure } of samples) stroke.points.push(x, y, width(stroke.tool, stroke.pointerType, pressure));
}

export function strokesOf(ink, key) {
  return ink.sections.get(key)?.strokes ?? [];
}

// Distance from (x, y) to the segment (ax, ay)-(bx, by).
function distance(x, y, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, length2 = dx * dx + dy * dy;
  const t = length2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / length2)) : 0;
  return Math.hypot(x - ax - t * dx, y - ay - t * dy);
}

function touches(stroke, x, y, radius) {
  const p = stroke.points;
  for (let i = 0; i < p.length; i += 3) {
    const j = Math.min(i + 3, p.length - 3);
    if (distance(x, y, p[i], p[i + 1], p[j], p[j + 1]) <= radius + p[i + 2] / 2) return true;
  }
  return false;
}

/** Removes the strokes passing within `radius` of (x, y); returns how many. */
export function erase(ink, key, x, y, radius) {
  const s = ink.sections.get(key);
  if (!s) return 0;
  const removed = [];
  s.strokes = s.strokes.filter((stroke, index) => {
    const hit = touches(stroke, x, y, radius);
    if (hit) removed.push({ stroke, index });
    return !hit;
  });
  if (removed.length) s.history.push({ type: 'erase', removed });
  return removed.length;
}

export function clear(ink, key) {
  const s = ink.sections.get(key);
  if (!s || !s.strokes.length) return;
  s.history.push({ type: 'clear', strokes: s.strokes });
  s.strokes = [];
}

export function undo(ink, key) {
  const s = ink.sections.get(key);
  const last = s?.history.pop();
  if (!last) return;
  if (last.type === 'add') s.strokes = s.strokes.filter(stroke => stroke !== last.stroke);
  else if (last.type === 'clear') s.strokes = last.strokes;
  else for (const { stroke, index } of last.removed) s.strokes.splice(index, 0, stroke);
}
