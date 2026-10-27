// Pixel algorithms on the class grid (Uint8Array W*H of base ids). No DOM here.
// Every write goes through set(), which enforces bounds and protect-source-class.

const shapeCache = new Map();
/** Spans [dy, dx0, dx1] (relative to the stamp origin) for a brush of `size`. */
function brushShape(size, round) {
  const key = size + (round ? "r" : "s");
  let s = shapeCache.get(key);
  if (s) return s;
  s = [];
  const r = size / 2;
  const lim = size === 1 ? 1 : r * r - size / 4; // n=1 single px, n=2 2x2, n=3 plus, ...
  for (let j = 0; j < size; j++) {
    let a = -1, b = -1;
    for (let i = 0; i < size; i++) {
      const ok = !round || size <= 2 || ((i + 0.5 - r) ** 2 + (j + 0.5 - r) ** 2 <= lim);
      if (ok) { if (a < 0) a = i; b = i; }
    }
    if (a >= 0) s.push([j, a, b]);
  }
  shapeCache.set(key, s);
  return s;
}

export class Painter {
  constructor() { this.W = 0; this.H = 0; this.grid = null; this.protect = -1; this.stack = null; this.resetBox(); }
  attach(grid, W, H) {
    this.grid = grid; this.W = W; this.H = H;
    if (!this.stack || this.stack.length !== W * H) this.stack = new Int32Array(W * H);
    this.resetBox();
  }
  resetBox() { this.x0 = 1e9; this.y0 = 1e9; this.x1 = -1; this.y1 = -1; }
  /** Returns the dirty rect {x,y,w,h} since the last take() (or null) and resets it. */
  take() {
    if (this.x1 < 0) return null;
    const r = { x: this.x0, y: this.y0, w: this.x1 - this.x0 + 1, h: this.y1 - this.y0 + 1 };
    this.resetBox();
    return r;
  }
  set(x, y, c) {
    if (x < 0 || y < 0 || x >= this.W || y >= this.H) return false;
    const i = y * this.W + x, g = this.grid[i];
    if (g === c || (this.protect >= 0 && g !== this.protect)) return false;
    this.grid[i] = c;
    if (x < this.x0) this.x0 = x; if (x > this.x1) this.x1 = x;
    if (y < this.y0) this.y0 = y; if (y > this.y1) this.y1 = y;
    return true;
  }
  /** Brush footprint anchored at pixel (cx, cy). A 1 px brush paints exactly that pixel. */
  stamp(cx, cy, size, round, c) {
    const o = (size - 1) >> 1;
    for (const [j, a, b] of brushShape(size, round)) {
      const y = cy - o + j;
      for (let i = a; i <= b; i++) this.set(cx - o + i, y, c);
    }
  }
  /** Bresenham from (x0,y0) to (x1,y1) inclusive, stamping at each step. */
  line(x0, y0, x1, y1, size, round, c) {
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    for (;;) {
      this.stamp(x0, y0, size, round, c);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }
  /** 4-connected flood relabel (or every pixel of that class when `global`). Returns pixels changed. */
  fillRegion(x, y, c, global = false) {
    const { W, H, grid, stack } = this;
    if (x < 0 || y < 0 || x >= W || y >= H) return 0;
    const target = grid[y * W + x];
    if (target === c || (this.protect >= 0 && target !== this.protect)) return 0;
    let n = 0;
    if (global) {
      for (let i = 0; i < grid.length; i++) if (grid[i] === target) { grid[i] = c; n++; this.touch(i % W, (i / W) | 0); }
      return n;
    }
    let sp = 0;
    grid[y * W + x] = c; stack[sp++] = y * W + x; n++;
    while (sp) {
      const i = stack[--sp], px = i % W, py = (i / W) | 0;
      this.touch(px, py);
      if (px > 0 && grid[i - 1] === target) { grid[i - 1] = c; stack[sp++] = i - 1; n++; }
      if (px < W - 1 && grid[i + 1] === target) { grid[i + 1] = c; stack[sp++] = i + 1; n++; }
      if (py > 0 && grid[i - W] === target) { grid[i - W] = c; stack[sp++] = i - W; n++; }
      if (py < H - 1 && grid[i + W] === target) { grid[i + W] = c; stack[sp++] = i + W; n++; }
    }
    return n;
  }
  touch(x, y) {
    if (x < this.x0) this.x0 = x; if (x > this.x1) this.x1 = x;
    if (y < this.y0) this.y0 = y; if (y > this.y1) this.y1 = y;
  }
  /** Scanline even-odd fill of polygon `pts` ([[x,y]...] in float image coords); pixel centres decide. */
  fillPolygon(pts, c) {
    const n = pts.length;
    if (n < 3) return 0;
    let minY = Infinity, maxY = -Infinity;
    for (const p of pts) { if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1]; }
    const ya = Math.max(0, Math.ceil(minY - 0.5)), yb = Math.min(this.H - 1, Math.ceil(maxY - 0.5) - 1);
    let changed = 0;
    const xs = [];
    for (let y = ya; y <= yb; y++) {
      const yc = y + 0.5;
      xs.length = 0;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const [x1, y1] = pts[j], [x2, y2] = pts[i];
        if ((y1 <= yc) !== (y2 <= yc)) xs.push(x1 + ((yc - y1) * (x2 - x1)) / (y2 - y1));
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(0, Math.ceil(xs[k] - 0.5)), xb = Math.min(this.W - 1, Math.ceil(xs[k + 1] - 0.5) - 1);
        for (let x = xa; x <= xb; x++) if (this.set(x, y, c)) changed++;
      }
    }
    return changed;
  }
  /** Paint every pixel where mask01[i] != 0 (SAM candidate). */
  applyMask(mask01, c) {
    const { W, grid } = this;
    let changed = 0;
    for (let i = 0; i < mask01.length; i++) {
      if (!mask01[i]) continue;
      if (this.set(i % W, (i / W) | 0, c)) changed++;
    }
    return changed;
  }
}
