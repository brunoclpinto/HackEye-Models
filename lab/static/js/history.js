// Per-frame undo/redo with bbox patches {x,y,w,h,before,after}. `shadow` mirrors the last committed grid,
// so a stroke needs no pre-copy: commit() diffs the touched rect against the shadow.

const MAX_PATCHES = 200, MAX_BYTES = 192 * 1024 * 1024;

export class History {
  constructor() { this.reset(null, 0, 0); }
  reset(grid, W, H) {
    this.W = W; this.H = H;
    this.shadow = grid ? grid.slice() : null;
    this.stack = []; this.pos = 0; this.bytes = 0;
    this.savedTop = null;
  }
  get top() { return this.pos ? this.stack[this.pos - 1] : null; }
  get dirty() { return this.top !== this.savedTop; }
  markSaved() { this.savedTop = this.top; }
  get canUndo() { return this.pos > 0; }
  get canRedo() { return this.pos < this.stack.length; }

  /** Record changes inside rect {x,y,w,h} of `grid`. Returns the tight patch or null if nothing differs. */
  commit(grid, rect, tool = "") {
    if (!rect) return null;
    const { W, shadow } = this;
    let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
    for (let y = rect.y; y < rect.y + rect.h; y++) {
      const row = y * W;
      for (let x = rect.x; x < rect.x + rect.w; x++) {
        if (grid[row + x] !== shadow[row + x]) {
          if (x < x0) x0 = x; if (x > x1) x1 = x;
          if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) return null;
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    const before = new Uint8Array(w * h), after = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const src = (y0 + y) * W + x0;
      before.set(shadow.subarray(src, src + w), y * w);
      after.set(grid.subarray(src, src + w), y * w);
      shadow.set(after.subarray(y * w, y * w + w), src);
    }
    const p = { x: x0, y: y0, w, h, before, after, tool };
    // drop redo branch
    for (let i = this.pos; i < this.stack.length; i++) this.bytes -= this.stack[i].before.length * 2;
    this.stack.length = this.pos;
    this.stack.push(p); this.pos++; this.bytes += w * h * 2;
    while (this.stack.length > MAX_PATCHES || this.bytes > MAX_BYTES) {
      const o = this.stack.shift(); this.bytes -= o.before.length * 2; this.pos--;
      if (this.savedTop === o) this.savedTop = undefined; // saved state fell off the stack: always dirty
    }
    return p;
  }
  _apply(grid, p, data) {
    const { W } = this;
    for (let y = 0; y < p.h; y++) {
      const dst = (p.y + y) * W + p.x, row = data.subarray(y * p.w, (y + 1) * p.w);
      grid.set(row, dst); this.shadow.set(row, dst);
    }
    return { x: p.x, y: p.y, w: p.w, h: p.h };
  }
  /** Returns the dirty rect or null. */
  undo(grid) { return this.canUndo ? this._apply(grid, this.stack[--this.pos], this.stack[this.pos].before) : null; }
  redo(grid) { return this.canRedo ? this._apply(grid, this.stack[this.pos++], this.stack[this.pos - 1].after) : null; }
}
