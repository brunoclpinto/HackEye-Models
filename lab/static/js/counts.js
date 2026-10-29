// Per-class detections of a class grid: one labelling pass (iterative flood fill, no recursion).
// A detection is a 4-connected region of one class (the same thing Inspect selects); no minimum size.
let seen = null, stack = null;

/** @returns {{[id: number]: {items: number, px: number}}} for every class present in the grid. */
export function classCounts(grid, W, H) {
  const n = W * H;
  if (!seen || seen.length !== n) { seen = new Uint8Array(n); stack = new Int32Array(n); } else seen.fill(0);
  const out = {};
  for (let s = 0; s < n; s++) {
    if (seen[s]) continue;
    const c = grid[s]; let sp = 0, px = 0;
    stack[sp++] = s; seen[s] = 1;
    while (sp) {
      const i = stack[--sp]; px++;
      const x = i % W;
      let j;
      if (x > 0 && !seen[j = i - 1] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (x < W - 1 && !seen[j = i + 1] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (i >= W && !seen[j = i - W] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
      if (i < n - W && !seen[j = i + W] && grid[j] === c) { seen[j] = 1; stack[sp++] = j; }
    }
    const o = out[c] || (out[c] = { items: 0, px: 0 });
    o.items++; o.px += px;
  }
  return out;
}
