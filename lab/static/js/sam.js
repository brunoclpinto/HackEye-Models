// SAM2 interaction state: points/box -> candidate masks (via /api/sam/predict) with Tab cycling.
import { api } from "./api.js";
import { toast } from "./util.js";

async function decodeMask(b64, W, H) {
  const img = new Image();
  img.src = "data:image/png;base64," + b64;
  await img.decode();
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const cx = c.getContext("2d", { willReadFrequently: true });
  cx.drawImage(img, 0, 0, W, H);
  const px = cx.getImageData(0, 0, W, H).data;
  const m = new Uint8Array(W * H);
  for (let i = 0; i < m.length; i++) m[i] = px[i * 4] ? 1 : 0;
  return m;
}

export class SamSession {
  constructor(onChange) {
    this.onChange = onChange; // called whenever visible state changes
    this.stem = null; this.W = 1280; this.H = 720;
    this.hatch = document.createElement("canvas");
    this.hatchKey = "";
    this.reset();
  }
  reset() {
    this.points = []; this.labels = []; this.box = null;
    this.cands = []; this.active = 0; this.seq = (this.seq || 0) + 1; this.busy = false;
    this.hatchKey = "";
    this.onChange?.();
  }
  setFrame(stem, W, H) { this.stem = stem; this.W = W; this.H = H; this.reset(); }
  prefetch(stem) { api("/api/sam/embed", { method: "POST", body: { stem }, quiet: true }).catch(() => {}); }
  get has() { return this.cands.length > 0; }
  get mask() { return this.cands[this.active]?.mask ?? null; }
  get info() { return this.cands.length ? `${this.active + 1}/${this.cands.length} score ${this.cands[this.active].score.toFixed(3)}` : (this.busy ? "thinking..." : ""); }

  addPoint(x, y, positive) { this.points.push([Math.floor(x), Math.floor(y)]); this.labels.push(positive ? 1 : 0); return this.predict(); }
  setBox(x1, y1, x2, y2) {
    this.box = [Math.floor(Math.min(x1, x2)), Math.floor(Math.min(y1, y2)), Math.ceil(Math.max(x1, x2)), Math.ceil(Math.max(y1, y2))];
    return this.predict();
  }
  cycle(dir = 1) {
    if (this.cands.length < 2) return;
    this.active = (this.active + dir + this.cands.length) % this.cands.length;
    this.hatchKey = ""; this.onChange?.();
  }
  async predict() {
    if (!this.points.length && !this.box) { this.cands = []; this.onChange?.(); return; }
    const seq = ++this.seq;
    this.busy = true; this.onChange?.();
    try {
      const r = await api("/api/sam/predict", { method: "POST", body: { stem: this.stem, points: this.points, labels: this.labels, box: this.box } });
      if (seq !== this.seq) return; // superseded by a newer click / frame change
      const cands = [];
      for (const m of r.masks || []) cands.push({ score: m.score, mask: await decodeMask(m.png_b64, this.W, this.H) });
      if (seq !== this.seq) return;
      this.cands = cands; this.active = 0; this.hatchKey = "";
      if (!cands.length) toast("SAM returned no mask", "error", 2500);
    } catch { /* api() already toasted */ }
    if (seq === this.seq) { this.busy = false; this.onChange?.(); }
  }
  /** Opaque-where-selected canvas (W x H) of the active candidate; cached until the candidate changes. */
  maskCanvas() {
    const m = this.mask;
    if (!m) return null;
    const key = `${this.active}|${this.seq}`;
    if (key === this.hatchKey) return this.hatch;
    const { W, H } = this;
    if (this.hatch.width !== W || this.hatch.height !== H) { this.hatch.width = W; this.hatch.height = H; }
    const cx = this.hatch.getContext("2d");
    const img = cx.createImageData(W, H), u = new Uint32Array(img.data.buffer);
    for (let i = 0; i < m.length; i++) if (m[i]) u[i] = 0xffffffff;
    cx.putImageData(img, 0, 0);
    this.hatchKey = key;
    return this.hatch;
  }
}
