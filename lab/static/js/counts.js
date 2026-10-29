// Per-class detections of a class grid: one labelling pass (iterative flood fill, no recursion).
// A detection is a 4-connected region of one class (the same thing Inspect selects); no minimum size.
let seen = null, stack = null;

/**
 * One pass over the grid. counts: {[id]: {items, px}}; regions: {[id]: [{k, px, bbox:[x0,y0,x1,y1] (inclusive), seed:[x,y]}]}
 * per class, sorted by px desc then seed (first pixel in scan order), numbered k = 1..n.
 */
export function classRegions(grid, W, H) {
  const n = W * H;
  if (!seen || seen.length !== n) { seen = new Uint8Array(n); stack = new Int32Array(n); } else seen.fill(0);
  const counts = {}, regions = {};
  for (let s = 0; s < n; s++) {
    if (seen[s]) continue;
    const c = grid[s]; let sp = 0, px = 0;
    const sy = (s / W) | 0, sx = s - sy * W;
    let x0 = sx, x1 = sx, y0 = sy, y1 = sy;
    stack[sp++] = s; seen[s] = 1;
    while (sp) {
      const i = stack[--sp]; px++;
      const y = (i / W) | 0, x = i - y * W;
      if (x < x0) x0 = x; else if (x > x1) x1 = x;
      if (y < y0) y0 = y; else if (y > y1) y1 = y;
      let j;
      if (x > 0 && !seen[j = i - 1] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (x < W - 1 && !seen[j = i + 1] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (i >= W && !seen[j = i - W] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (i < n - W && !seen[j = i + W] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
    }
    const o = counts[c] || (counts[c] = { items: 0, px: 0 });
    o.items++; o.px += px;
    (regions[c] || (regions[c] = [])).push({ k: 0, px, bbox: [x0, y0, x1, y1], seed: [sx, sy], s });
  }
  for (const list of Object.values(regions)) {
    list.sort((a, b) => b.px - a.px || a.s - b.s);          // equal sizes: scan order of the seed
    list.forEach((r, i) => { r.k = i + 1; delete r.s; });
  }
  return { counts, regions };
}

/** @returns {{[id: number]: {items: number, px: number}}} for every class present in the grid. */
export const classCounts = (grid, W, H) => classRegions(grid, W, H).counts;
