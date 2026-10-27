// Mask editor: offscreen image + mask layers, one visible canvas (setTransform, no smoothing), rAF-batched redraws.
import { api } from "./api.js";
import { $, $$, el, toast, clamp, status, rgbCss, confirmDialog, debounce } from "./util.js";
import { S, clsName, clsColor, samAvailable, loadMeta } from "./store.js";
import { Painter } from "./tools.js";
import { History } from "./history.js";
import { Picker } from "./picker.js";
import { SamSession } from "./sam.js";
import { hashFor } from "./nav.js";

const MAXZ = 32, TOOL_NAMES = { inspect: "Inspect", brush: "Brush", erase: "Erase", region: "Region", polygon: "Lasso", sam: "SAM" };
const painter = new Painter(), history = new History();

const E = {
  stem: null, meta: null, qs: "", W: 1280, H: 720, grid: null,
  tool: "inspect", sel: null, selCanvas: null, samNeg: false, locked: false, saveTool: "inspect", gesture: null, lastTap: null, size: 8, round: true, showMask: true, outline: false, showGrid: true, opacity: 0.55,
  z: 1, vx: 0, vy: 0, fitZ: 1, fitted: true, cw: 0, ch: 0, dpr: 1,
  cursor: null, drag: null, space: false, poly: null,
  used: new Set(), loadSeq: 0, saving: null,
  viewMapName: "", viewLut: identity(), lut32: new Uint32Array(256),
  cache: new Map(),
};
function identity() { const a = new Uint8Array(256); for (let i = 0; i < 256; i++) a[i] = i; return a; }

const imgC = document.createElement("canvas"), maskC = document.createElement("canvas");
let imgCtx, maskCtx, maskImg, mask32, canvas, ctx, picker, sam, raf = 0, strokeRect = null;

// ---------------------------------------------------------------- init
export function init() {
  canvas = $("#eCanvas"); ctx = canvas.getContext("2d", { alpha: false });
  picker = new Picker($("#ePicker"), $("#eProtect"), () => { painter.protect = picker.protect; syncUI(); redraw(); });
  sam = new SamSession(() => { syncUI(); redraw(); });
  new ResizeObserver(() => onResize()).observe($("#eStage"));
  bindToolbar(); bindCanvas();
  buildLut(); syncUI();
  window.addEventListener("beforeunload", (e) => { if (history.dirty) { e.preventDefault(); e.returnValue = ""; } });
}
window.__lab = {
  state: () => ({ stem: E.stem, zoom: E.z, tool: E.tool, cls: picker?.active, dirty: history.dirty, samCandidates: sam ? sam.cands.length : 0,
    selection: E.sel ? { id: E.sel.id, px: E.sel.n, bbox: [...E.sel.bbox], label: selText() } : null }),
  imgToClient: (x, y) => { const r = canvas.getBoundingClientRect(); return { x: r.left + E.vx + (x + 0.5) * E.z, y: r.top + E.vy + (y + 0.5) * E.z }; },
  clientToImg: (cx, cy) => { const r = canvas.getBoundingClientRect(), fx = (cx - r.left - E.vx) / E.z, fy = (cy - r.top - E.vy) / E.z; return { x: Math.floor(fx), y: Math.floor(fy), fx, fy }; },
  getPixel: (x, y) => (E.grid ? E.grid[y * E.W + x] : undefined),
};
export const currentStem = () => E.stem;
export const currentQS = () => E.qs;
export const isDirty = () => history.dirty;

// ---------------------------------------------------------------- colour LUT / mask layer
function buildLut() {
  const l = E.lut32;
  const put = (id, c) => { l[id] = (255 << 24 | c[2] << 16 | c[1] << 8 | c[0]) >>> 0; };
  for (let i = 0; i < 256; i++) put(i, [70, 70, 70]);
  if (E.viewMap) { for (const c of E.viewMap.classes) put(c.id, c.void ? [60, 60, 60] : c.color); }
  else { for (const c of S.tax) put(c.id, c.color); put(255, [60, 60, 60]); }
}
function paintRect(rx, ry, rw, rh) {
  if (!E.grid) return;
  const { W, H, grid } = E, vl = E.viewLut, lut = E.lut32;
  if (E.outline) { rx--; ry--; rw++; rh++; }
  const x0 = Math.max(0, rx), y0 = Math.max(0, ry), x1 = Math.min(W, rx + rw), y1 = Math.min(H, ry + rh);
  if (x1 <= x0 || y1 <= y0) return;
  for (let y = y0; y < y1; y++) {
    let i = y * W + x0;
    for (let x = x0; x < x1; x++, i++) {
      const d = vl[grid[i]];
      if (E.outline) {
        const r = x < W - 1 && vl[grid[i + 1]] !== d, b = y < H - 1 && vl[grid[i + W]] !== d;
        mask32[i] = r || b ? lut[d] : 0;
      } else mask32[i] = lut[d];
    }
  }
  maskCtx.putImageData(maskImg, 0, 0, x0, y0, x1 - x0, y1 - y0);
  redraw();
}

// ---------------------------------------------------------------- view / drawing
function onResize() {
  const st = $("#eStage"), r = st.getBoundingClientRect();
  if (r.width < 10 || r.height < 10) return;
  E.dpr = window.devicePixelRatio || 1;
  E.cw = r.width; E.ch = r.height;
  canvas.width = Math.round(r.width * E.dpr); canvas.height = Math.round(r.height * E.dpr);
  E.fitZ = Math.min(E.cw / E.W, E.ch / E.H);
  if (E.fitted) fit(); else { E.z = clamp(E.z, E.fitZ, MAXZ); clampPan(); }
  redraw();
}
export const onShow = () => requestAnimationFrame(onResize);
function fit() {
  E.fitZ = Math.min(E.cw / E.W, E.ch / E.H);
  E.z = E.fitZ; E.vx = (E.cw - E.W * E.z) / 2; E.vy = (E.ch - E.H * E.z) / 2; E.fitted = true; redraw(); syncStatus();
}
function clampPan() {
  const m = 64;
  E.vx = clamp(E.vx, m - E.W * E.z, E.cw - m); E.vy = clamp(E.vy, m - E.H * E.z, E.ch - m);
}
function zoomAt(sx, sy, nz) {
  nz = clamp(nz, E.fitZ, MAXZ);
  const k = nz / E.z;
  E.vx = sx - (sx - E.vx) * k; E.vy = sy - (sy - E.vy) * k; E.z = nz; E.fitted = Math.abs(nz - E.fitZ) < 1e-6;
  clampPan(); redraw(); syncStatus();
}
let hatchTmp = null, hatchPat = { key: "", pat: null };
/** Hatched highlight of the active SAM candidate, in screen space so stripes keep their size at any zoom. */
function drawHatch() {
  const m = sam.maskCanvas();
  if (!m) return;
  if (!hatchTmp) hatchTmp = document.createElement("canvas");
  if (hatchTmp.width !== canvas.width || hatchTmp.height !== canvas.height) { hatchTmp.width = canvas.width; hatchTmp.height = canvas.height; }
  const t = hatchTmp.getContext("2d"), { dpr, z, vx, vy } = E, rgb = clsColor(picker.active);
  t.globalCompositeOperation = "source-over"; t.setTransform(1, 0, 0, 1, 0, 0); t.clearRect(0, 0, hatchTmp.width, hatchTmp.height);
  t.setTransform(z * dpr, 0, 0, z * dpr, vx * dpr, vy * dpr); t.imageSmoothingEnabled = false; t.drawImage(m, 0, 0);
  t.setTransform(1, 0, 0, 1, 0, 0); t.globalCompositeOperation = "source-in";
  const key = `${rgb}|${dpr}`;
  if (hatchPat.key !== key) {
    const n = Math.round(12 * dpr), tile = document.createElement("canvas"); tile.width = tile.height = n;
    const c = tile.getContext("2d"); c.fillStyle = rgbCss(rgb, 0.28); c.fillRect(0, 0, n, n);
    c.strokeStyle = rgbCss(rgb, 0.95); c.lineWidth = 3 * dpr; c.beginPath();
    for (const o of [-n, 0, n]) { c.moveTo(o, n); c.lineTo(o + n, 0); }
    c.stroke();
    hatchPat = { key, pat: t.createPattern(tile, "repeat") };
  }
  t.fillStyle = hatchPat.pat; t.fillRect(0, 0, hatchTmp.width, hatchTmp.height);
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(hatchTmp, 0, 0);
  ctx.setTransform(z * dpr, 0, 0, z * dpr, vx * dpr, vy * dpr);
}
// ---------------------------------------------------------------- selection (Inspect tool)
function selColor() {
  const id = E.sel.id;
  if (E.viewMap) { const m = E.viewMap.classes.find((c) => c.id === E.viewLut[id]); if (m) return m.void ? [90, 90, 90] : m.color; }
  return clsColor(id);
}
function selText() {
  if (!E.sel) return "";
  let t = `${clsName(E.sel.id)} \u00b7 id ${E.sel.id} \u00b7 ${E.sel.n} px`;
  if (E.viewMap) { const m = E.viewMap.classes.find((c) => c.id === E.viewLut[E.sel.id]); if (m) t += ` \u00b7 ${m.name}`; }
  return t;
}
function setSel(r) {
  E.sel = r;
  if (r) {
    const c = selColor(), u = new Uint32Array(E.W * E.H), v = ((38 << 24) | (c[2] << 16) | (c[1] << 8) | c[0]) >>> 0;
    for (let i = 0; i < u.length; i++) if (r.mask[i]) u[i] = v;
    if (!E.selCanvas) E.selCanvas = document.createElement("canvas");
    E.selCanvas.width = E.W; E.selCanvas.height = E.H;
    const cx = E.selCanvas.getContext("2d"), img = cx.createImageData(E.W, E.H);
    new Uint32Array(img.data.buffer).set(u); cx.putImageData(img, 0, 0);
  }
  syncSel(); redraw();
}
const clearSel = () => { if (E.sel) setSel(null); };
function syncSel() {
  const l = $("#selLabel"), t = selText();
  l.hidden = !t; l.textContent = t;
  syncToolStatus();
}
/** HackEye-Lab style box + label tag (screen space, crisp at any zoom). */
function drawSel(sx, sy) {
  const [x0, y0, x1, y1] = E.sel.bbox, col = selColor(), css = rgbCss(col);
  const X = sx(x0), Y = sy(y0), w = (x1 - x0 + 1) * E.z, h = (y1 - y0 + 1) * E.z;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineWidth = 5; ctx.strokeStyle = "rgba(0,0,0,.55)"; ctx.strokeRect(X, Y, w, h);
  ctx.lineWidth = 3; ctx.strokeStyle = css; ctx.strokeRect(X, Y, w, h);
  const text = selText();
  ctx.font = "600 13px system-ui, sans-serif"; ctx.textBaseline = "middle";
  const pad = 4, th = 18, tw = ctx.measureText(text).width + pad * 2;
  const lx = Math.min(Math.max(0, X - 1.5), Math.max(0, E.cw - tw));
  let ly = Y - th - 1.5;
  if (ly < 0) ly = Y + 1.5;                                 // no room above: label inside the box
  ly = Math.min(ly, Math.max(0, E.ch - th));
  ctx.fillStyle = css; ctx.fillRect(lx, ly, tw, th);
  const lum = (0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]) / 255;
  ctx.fillStyle = lum > 0.55 ? "#111" : "#fff"; ctx.fillText(text, lx + pad, ly + th / 2 + 0.5);
  ctx.restore();
}
function redraw() { if (!raf) raf = requestAnimationFrame(draw); }

function draw() {
  raf = 0;
  if (!ctx || !E.cw) return;
  const { dpr, z, vx, vy } = E;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#050607"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!E.grid) return;
  ctx.setTransform(z * dpr, 0, 0, z * dpr, vx * dpr, vy * dpr);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(imgC, 0, 0);
  if (E.showMask) { ctx.globalAlpha = E.outline ? Math.max(E.opacity, 0.85) : E.opacity; ctx.drawImage(maskC, 0, 0); ctx.globalAlpha = 1; }
  if (E.tool === "sam" && sam.has) drawHatch();
  if (E.sel && E.selCanvas) ctx.drawImage(E.selCanvas, 0, 0);
  // screen-space overlays
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sx = (ix) => vx + ix * z, sy = (iy) => vy + iy * z;
  if (E.showGrid && z >= 8) {
    ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1; ctx.beginPath();
    const xa = Math.max(0, Math.floor(-vx / z)), xb = Math.min(E.W, Math.ceil((E.cw - vx) / z));
    const ya = Math.max(0, Math.floor(-vy / z)), yb = Math.min(E.H, Math.ceil((E.ch - vy) / z));
    for (let x = xa; x <= xb; x++) { const px = Math.round(sx(x)) + 0.5; ctx.moveTo(px, Math.max(0, sy(ya))); ctx.lineTo(px, Math.min(E.ch, sy(yb))); }
    for (let y = ya; y <= yb; y++) { const py = Math.round(sy(y)) + 0.5; ctx.moveTo(Math.max(0, sx(xa)), py); ctx.lineTo(Math.min(E.cw, sx(xb)), py); }
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = 1; ctx.strokeRect(sx(0) - 0.5, sy(0) - 0.5, E.W * z + 1, E.H * z + 1);
  if (E.sel) drawSel(sx, sy);
  const ta = E.tool === "inspect" && E.fitted ? "pan-y" : "none";   // fit + Inspect: the page scrolls natively on touch
  if (canvas.style.touchAction !== ta) canvas.style.touchAction = ta;
  // polygon in progress
  if (E.poly && E.poly.length) {
    const pts = E.poly, col = clsColor(picker.active);
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))));
    if (E.cursor && !E.drag) ctx.lineTo(sx(E.cursor.fx), sy(E.cursor.fy));
    ctx.fillStyle = rgbCss(col, 0.22); ctx.fill("evenodd");
    ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = "#fff"; for (const p of pts) ctx.fillRect(sx(p[0]) - 2, sy(p[1]) - 2, 4, 4);
  }
  // SAM prompts
  if (E.tool === "sam") {
    sam.points.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(sx(p[0] + 0.5), sy(p[1] + 0.5), 5, 0, 7);
      ctx.fillStyle = sam.labels[i] ? "#34d399" : "#ff5d5d"; ctx.fill(); ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5; ctx.stroke();
    });
    const b = sam.box || (E.drag?.kind === "sam" && E.drag.box);
    if (b) { ctx.strokeStyle = "#22d3ee"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.strokeRect(sx(b[0]), sy(b[1]), (b[2] - b[0]) * z, (b[3] - b[1]) * z); ctx.setLineDash([]); }
  }
  // brush footprint
  if (E.cursor && (E.tool === "brush" || E.tool === "erase") && !E.space && !E.drag?.pan) {
    const { px, py } = E.cursor, n = E.size, o = (n - 1) >> 1;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.shadowColor = "#000"; ctx.shadowBlur = 2;
    if (E.round && n > 2) { ctx.beginPath(); ctx.ellipse(sx(px - o + n / 2), sy(py - o + n / 2), n * z / 2, n * z / 2, 0, 0, 7); ctx.stroke(); }
    else ctx.strokeRect(sx(px - o) + 0.5, sy(py - o) + 0.5, n * z - 1, n * z - 1);
    ctx.shadowBlur = 0;
  } else if (E.cursor && E.tool === "region" && z >= 4) {
    ctx.strokeStyle = "#fff"; ctx.strokeRect(sx(E.cursor.px) + 0.5, sy(E.cursor.py) + 0.5, z - 1, z - 1);
  }
}

// ---------------------------------------------------------------- pointer input
const ptr = (e) => {
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  const fx = (sx - E.vx) / E.z, fy = (sy - E.vy) / E.z;
  return { sx, sy, fx, fy, px: Math.floor(fx), py: Math.floor(fy) };
};
const inside = (p) => p.px >= 0 && p.py >= 0 && p.px < E.W && p.py < E.H;

const touches = new Map();           // pointerId -> {x, y} (canvas coords) of the touch pointers currently down
function bindCanvas() {
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const p = ptr(e), k = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 16 : 1) * (e.ctrlKey ? 0.01 : 0.0015));
    zoomAt(p.sx, p.sy, E.z * k);
  }, { passive: false });
  for (const t of ["gesturestart", "gesturechange", "gestureend"]) canvas.addEventListener(t, (e) => e.preventDefault());   // WebKit page zoom
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("pointerleave", () => { if (!E.drag) { E.cursor = null; status.set("cursor", ""); redraw(); } });
  canvas.addEventListener("dblclick", () => { if (E.tool === "polygon" && E.poly) closePoly(); });
}

/** Undo the pixels of a started stroke (second finger landed / browser took the gesture). */
function rollback() {
  if (strokeRect) {
    const { x, y, w, h } = strokeRect, W = E.W;
    for (let r = 0; r < h; r++) { const o = (y + r) * W + x; E.grid.set(history.shadow.subarray(o, o + w), o); }
    paintRect(x, y, w, h);
  }
  strokeRect = null; painter.resetBox();
}
function cancelAction() {
  if (E.drag?.kind === "stroke") rollback();
  if (E.drag?.kind === "poly" && E.drag.free) E.poly = null;
  E.drag = null; redraw();
}
function startGesture() {
  cancelAction();
  const [a, b] = [...touches.values()], mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  E.gesture = { d0: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), z0: E.z, fx0: (mx - E.vx) / E.z, fy0: (my - E.vy) / E.z };
  canvas.classList.add("panning");
}
/** Pinch zoom around the midpoint + two-finger pan: the image point that was under the start midpoint follows the midpoint. */
function applyGesture() {
  const [a, b] = [...touches.values()], g = E.gesture;
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, d = Math.hypot(a.x - b.x, a.y - b.y);
  E.z = clamp(g.z0 * d / g.d0, E.fitZ, MAXZ);
  E.vx = mx - g.fx0 * E.z; E.vy = my - g.fy0 * E.z;
  E.fitted = Math.abs(E.z - E.fitZ) < 1e-6; clampPan(); redraw(); syncStatus();
}

function onDown(e) {
  if (!E.grid) return;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic / released pointer */ }
  e.preventDefault();
  const p = ptr(e), touch = e.pointerType === "touch";
  if (touch) {
    touches.set(e.pointerId, { x: p.sx, y: p.sy });
    if (touches.size >= 2) { if (!E.gesture) startGesture(); return; }
    if (E.gesture) return;
  }
  if (e.button === 1 || (e.button === 0 && E.space)) { E.drag = { pan: true, sx: p.sx, sy: p.sy, vx: E.vx, vy: E.vy }; canvas.classList.add("panning"); return; }
  if (E.tool === "inspect") {
    if (e.button !== 0) return;
    if (e.altKey && inside(p)) { altPick(e, p); return; }
    if (touch) E.drag = { kind: "tap", sx: p.sx, sy: p.sy, vx: E.vx, vy: E.vy, moved: false };
    else if (inside(p)) setSel(painter.select(p.px, p.py));
    return;
  }
  if (!inside(p) && E.tool !== "sam" && E.tool !== "polygon") return;
  if (e.button === 0 && e.altKey && inside(p)) { altPick(e, p); return; }
  const c = E.tool === "erase" ? picker.secondary : picker.active;
  if (e.button === 0 && (E.tool === "brush" || E.tool === "erase")) {
    E.drag = { kind: "stroke", lx: p.px, ly: p.py, c }; strokeRect = null; painter.protect = picker.protect;
    painter.stamp(p.px, p.py, E.size, E.round, c); flushPaint();
  } else if (e.button === 0 && E.tool === "region") {
    if (touch) { E.drag = { kind: "tap", sx: p.sx, sy: p.sy, moved: false, region: true, shift: false }; return; }   // applied on lift, so a 2nd finger can cancel
    painter.protect = picker.protect;
    painter.fillRegion(p.px, p.py, picker.active, e.shiftKey);
    strokeRect = null; flushPaint(); endEdit("region");
  } else if (E.tool === "polygon" && e.button === 0) {
    if (touch) { E.poly = [[p.fx, p.fy]]; E.drag = { kind: "poly", lx: p.sx, ly: p.sy, free: true }; redraw(); return; }   // touch lasso = freehand drag
    E.poly = E.poly || [];
    E.poly.push([p.fx, p.fy]); E.drag = { kind: "poly", lx: p.sx, ly: p.sy }; redraw();
  } else if (E.tool === "sam" && (e.button === 0 || e.button === 2)) {
    E.drag = { kind: "sam", btn: e.button, sx: p.sx, sy: p.sy, fx: p.fx, fy: p.fy, box: null };
  }
}
function altPick(e, p) {            // eyedropper / protect-from-pixel
  const id = E.grid[p.py * E.W + p.px];
  if (e.shiftKey) { picker.setProtect(id); toast(`Protect: ${clsName(id)}`, "ok", 1500); } else picker.pick(id);
}

function onMove(e) {
  if (e.pointerType === "touch" && touches.has(e.pointerId)) {
    const p = ptr(e); touches.set(e.pointerId, { x: p.sx, y: p.sy });
    if (E.gesture) { applyGesture(); return; }
  }
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  if (!evs.length) evs.push(e);
  for (const ev of evs) handleMove(ev);
  flushPaint();
}
function handleMove(e) {
  const p = ptr(e);
  E.cursor = p;
  const d = E.drag;
  if (d?.pan) { E.vx = d.vx + (p.sx - d.sx); E.vy = d.vy + (p.sy - d.sy); clampPan(); E.fitted = false; redraw(); syncStatus(); }
  else if (d?.kind === "tap") {
    if (!d.moved && Math.hypot(p.sx - d.sx, p.sy - d.sy) > 8) d.moved = true;
    if (d.moved && !d.region && !E.fitted) { E.vx = d.vx + (p.sx - d.sx); E.vy = d.vy + (p.sy - d.sy); clampPan(); redraw(); syncStatus(); }   // zoomed Inspect: drag pans
  }
  else if (d?.kind === "stroke") { painter.line(d.lx, d.ly, p.px, p.py, E.size, E.round, d.c); d.lx = p.px; d.ly = p.py; }
  else if (d?.kind === "poly") {
    if (Math.hypot(p.sx - d.lx, p.sy - d.ly) >= 4) { E.poly.push([p.fx, p.fy]); d.lx = p.sx; d.ly = p.sy; redraw(); }
  } else if (d?.kind === "sam") {
    if (d.btn === 0 && !E.samNeg && Math.hypot(p.sx - d.sx, p.sy - d.sy) > 5) {
      d.box = [Math.min(d.fx, p.fx), Math.min(d.fy, p.fy), Math.max(d.fx, p.fx), Math.max(d.fy, p.fy)]; redraw();
    }
  } else redraw();
  status.set("cursor", inside(p) ? cursorText(p) : "");
}
function cursorText(p) {
  const id = E.grid ? E.grid[p.py * E.W + p.px] : 0;
  let t = `x ${p.px}  y ${p.py}  ${id} ${clsName(id)}`;
  if (E.viewMap) { const m = E.viewMap.classes.find((c) => c.id === E.viewLut[id]); if (m) t += `  -> ${m.name}`; }
  return t;
}
function onUp(e) {
  const cancel = e.type === "pointercancel";
  if (e.pointerType === "touch") {
    touches.delete(e.pointerId);
    if (E.gesture) { if (!touches.size) { E.gesture = null; canvas.classList.remove("panning"); } return; }
  }
  const d = E.drag;
  try { canvas.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
  if (!d) return;
  E.drag = null; canvas.classList.remove("panning");
  if (d.kind === "stroke") { if (cancel) rollback(); else endEdit(E.tool === "erase" ? "brush" : E.tool); }
  else if (d.kind === "poly" && d.free) { if (cancel) E.poly = null; else closePoly(); }
  else if (d.kind === "tap") { if (!cancel && !d.moved) tap(d); }
  else if (d.kind === "sam") {
    if (d.box) sam.setBox(...d.box);
    else sam.addPoint(d.fx, d.fy, d.btn === 0 && !E.samNeg);
  }
  redraw();
}
/** A finished one-finger tap: select (Inspect, double-tap toggles fit/4x) or apply the Region fill. */
function tap(d) {
  const fx = (d.sx - E.vx) / E.z, fy = (d.sy - E.vy) / E.z, px = Math.floor(fx), py = Math.floor(fy);
  if (d.region) {
    if (px < 0 || py < 0 || px >= E.W || py >= E.H) return;
    painter.protect = picker.protect; painter.fillRegion(px, py, picker.active, false); strokeRect = null; flushPaint(); endEdit("region");
    return;
  }
  const now = performance.now(), lt = E.lastTap;
  if (lt && now - lt.t < 320 && Math.hypot(lt.x - d.sx, lt.y - d.sy) < 30) {
    E.lastTap = null;
    if (E.fitted) zoomAt(d.sx, d.sy, 4); else fit();
    return;
  }
  E.lastTap = { t: now, x: d.sx, y: d.sy };
  if (px >= 0 && py >= 0 && px < E.W && py < E.H) setSel(painter.select(px, py));
}

/** Repaint whatever the painter has touched and grow the stroke's history rect. */
function flushPaint() {
  const r = painter.take();
  if (!r) return;
  paintRect(r.x, r.y, r.w, r.h);
  if (!strokeRect) strokeRect = { ...r };
  else {
    const x0 = Math.min(strokeRect.x, r.x), y0 = Math.min(strokeRect.y, r.y);
    const x1 = Math.max(strokeRect.x + strokeRect.w, r.x + r.w), y1 = Math.max(strokeRect.y + strokeRect.h, r.y + r.h);
    strokeRect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }
}
function endEdit(tool) {
  clearSel();
  const p = history.commit(E.grid, strokeRect, tool);
  strokeRect = null;
  if (p) E.used.add(tool);
  syncUI();
}
function closePoly() {
  const pts = E.poly; E.poly = null;
  if (pts && pts.length >= 3) { painter.protect = picker.protect; painter.fillPolygon(pts, picker.active); strokeRect = null; flushPaint(); endEdit("polygon"); }
  redraw();
}
function applySam() {
  if (!sam.has) return;
  painter.protect = picker.protect;
  painter.applyMask(sam.mask, picker.active);
  strokeRect = null; flushPaint(); endEdit("sam"); sam.reset();
}
/** Enter in Inspect: relabel the selected region to the active class (protect respected), one undoable patch. */
function relabelSel() {
  const r = E.sel; if (!r) return;
  painter.protect = picker.protect; painter.applyMask(r.mask, picker.active);
  strokeRect = null; flushPaint(); endEdit("region");
}
function undoRedo(redo) {
  clearSel();
  const r = redo ? history.redo(E.grid) : history.undo(E.grid);
  if (r) paintRect(r.x, r.y, r.w, r.h);
  syncUI();
}

// ---------------------------------------------------------------- load / save
function loadImage(url) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("image load failed: " + url)); i.src = url; });
}
async function decodeGrid(blob) {
  let bmp;
  try { bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" }); }
  catch { bmp = await createImageBitmap(blob); }
  const c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
  const cx = c.getContext("2d", { willReadFrequently: true }); cx.drawImage(bmp, 0, 0);
  const px = cx.getImageData(0, 0, c.width, c.height).data, grid = new Uint8Array(c.width * c.height);
  for (let i = 0; i < grid.length; i++) grid[i] = px[i * 4];            // red channel = base id
  return { grid, W: c.width, H: c.height };
}
function assets(stem) {
  let a = E.cache.get(stem);
  if (!a) {
    a = Promise.all([
      loadImage(`/api/frames/${encodeURIComponent(stem)}/image.jpg`),
      fetch(`/api/frames/${encodeURIComponent(stem)}/mask.png`, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`mask ${r.status}`); return r.blob(); }).then(decodeGrid),
    ]).then(([img, g]) => ({ img, ...g }));
    a.catch(() => E.cache.delete(stem));
    E.cache.set(stem, a);
    while (E.cache.size > 4) E.cache.delete(E.cache.keys().next().value);
  }
  return a;
}
function preload(stem) { if (stem && stem !== E.stem) assets(stem).catch(() => {}); }

/** Open a frame. qs = browse filter query string (drives prev/next). */
export async function show(stem, qs = "", { discard = false } = {}) {
  if (!discard && E.stem && history.dirty) {
    if (!(await save())) { if (!(await confirmDialog("Saving failed. Discard unsaved edits?", "Discard"))) { window.history.replaceState(null, "", hashFor(E.stem, undefined, E.qs)); return; } }
  }
  const seq = ++E.loadSeq;
  $("#eLoading").hidden = false;
  try {
    if (discard) E.cache.delete(stem);
    const [meta, a] = await Promise.all([
      api(`/api/frames/${encodeURIComponent(stem)}${qs ? "?" + qs : ""}`).catch(() => api(`/api/frames/${encodeURIComponent(stem)}`)),
      assets(stem),
    ]);
    if (seq !== E.loadSeq) return;
    E.cache.delete(stem);                       // the grid is mutated in place; never reuse it
    E.stem = stem; E.qs = qs; E.meta = meta; E.W = a.W; E.H = a.H; E.grid = a.grid;
    imgC.width = maskC.width = a.W; imgC.height = maskC.height = a.H;
    imgCtx = imgC.getContext("2d"); maskCtx = maskC.getContext("2d");
    imgCtx.drawImage(a.img, 0, 0, a.W, a.H);
    maskImg = maskCtx.createImageData(a.W, a.H); mask32 = new Uint32Array(maskImg.data.buffer);
    painter.attach(a.grid, a.W, a.H); painter.protect = picker.protect;
    clearSel(); history.reset(a.grid, a.W, a.H); E.used.clear(); E.poly = null; strokeRect = null;
    sam.setFrame(stem, a.W, a.H);
    buildLut(); paintRect(0, 0, a.W, a.H);
    { const pr = new Set(); const seen = new Uint8Array(256); for (let i = 0; i < a.grid.length; i++) seen[a.grid[i]] = 1; seen.forEach((v, id) => v && pr.add(id)); picker.setPresent(pr); }
    E.fitted = true; onResize();
    syncUI();
    if (samAvailable()) sam.prefetch(stem);
    preload(meta.next);
    if (meta.next) api(`/api/frames/${encodeURIComponent(meta.next)}${qs ? "?" + qs : ""}`, { quiet: true }).then((m) => preload(m.next)).catch(() => {});
    refreshMaps();
    document.dispatchEvent(new CustomEvent("frame-opened", { detail: { stem, index: meta.index, total: meta.total } }));
  } catch (e) {
    toast(`Could not open ${stem}: ${e.message}`);
  } finally { if (seq === E.loadSeq) $("#eLoading").hidden = true; }
}

function encodePng() {
  const { W, H, grid } = E, c = document.createElement("canvas"); c.width = W; c.height = H;
  const cx = c.getContext("2d"), img = cx.createImageData(W, H), u = img.data;
  for (let i = 0, j = 0; i < grid.length; i++, j += 4) { const v = grid[i]; u[j] = v; u[j + 1] = v; u[j + 2] = v; u[j + 3] = 255; }
  cx.putImageData(img, 0, 0);
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("PNG encode failed"))), "image/png"));
}

/** Save if dirty. Resolves true on success / nothing to do. */
export async function save() {
  if (E.saving) await E.saving.catch(() => {});
  if (!E.grid || !history.dirty) return true;
  const stem = E.stem, top = history.top, tool = E.used.size === 1 ? [...E.used][0] : "mixed";
  const run = (async () => {
    const blob = await encodePng();
    const res = await fetch(`/api/frames/${encodeURIComponent(stem)}/mask?tool=${tool}`, { method: "PUT", headers: { "Content-Type": "image/png" }, body: blob });
    if (!res.ok) { let d = `${res.status}`; try { d = (await res.json()).detail ?? d; } catch { /* */ } throw new Error(typeof d === "string" ? d : JSON.stringify(d)); }
    return res.json();
  })();
  E.saving = run;
  try {
    const r = await run;
    if (stem === E.stem) { history.savedTop = top; E.used.clear(); if (E.meta) { E.meta.status = r.status ?? E.meta.status; E.meta.edit_count = r.edit_count ?? E.meta.edit_count; } }
    document.dispatchEvent(new CustomEvent("frame-updated", { detail: { stem, status: r.status, edit_count: r.edit_count } }));
    toast(`Saved ${stem}: ${r.changed_px} px changed`, "ok", 1800);
    syncUI();
    return true;
  } catch (e) { toast(`Save failed: ${e.message}`); return false; }
  finally { E.saving = null; }
}
export async function leave() { return history.dirty ? save() : true; }

async function revert() {
  if (!E.stem) return;
  if (!window.confirm(`Revert ${E.stem} to its original mask? Saved edits are discarded.`)) return;
  try {
    await api(`/api/frames/${encodeURIComponent(E.stem)}/revert`, { method: "POST" });
    document.dispatchEvent(new CustomEvent("frame-updated", { detail: { stem: E.stem, status: "todo", edit_count: 0 } }));
    await show(E.stem, E.qs, { discard: true });
  } catch { /* toasted */ }
}
export async function setStatus(st) {
  if (!E.stem) return;
  try {
    await api(`/api/frames/${encodeURIComponent(E.stem)}/state`, { method: "PUT", body: { status: st } });
    E.meta.status = st; syncUI();
    document.dispatchEvent(new CustomEvent("frame-updated", { detail: { stem: E.stem, status: st } }));
  } catch { /* toasted */ }
}
async function go(stem) {
  if (!stem) { toast("No more frames in this filter", "ok", 1500); return; }
  if (!(await leave())) return;
  location.hash = hashFor(stem, undefined, E.qs);
}

// ---------------------------------------------------------------- view-through-map
async function refreshMaps() {
  try {
    const maps = await api("/api/maps", { quiet: true }), sel = $("#eMap"), cur = sel.value;
    sel.replaceChildren(el("option", { value: "" }, "none"), ...maps.map((m) => el("option", { value: m.name }, `${m.name} (${m.n_classes})`)));
    sel.value = [...sel.options].some((o) => o.value === cur) ? cur : "";
  } catch { /* maps optional */ }
}
function applyView(pv) {
  E.viewMap = pv || null;
  E.viewLut = pv ? Uint8Array.from(pv.lut.map((v) => clamp(v, 0, 255))) : identity();
  buildLut(); paintRect(0, 0, E.W, E.H);
  if (E.sel) setSel(E.sel);
}
async function setViewMap(name) {
  E.viewMapName = name; E.mapPv = null;
  if (name) {
    try {
      const m = await api(`/api/maps/${encodeURIComponent(name)}`);
      E.mapPv = await api("/api/maps/preview", { method: "POST", body: { groups: m.groups } });
    } catch { $("#eMap").value = ""; }
  }
  if (!E.override) applyView(E.mapPv);
}
/** The Taxonomy tab pushes its live map preview here; null restores the Edit tab's selected map. */
export function setViewPreview(pv) {
  if (!pv && !E.override) return;
  E.override = pv || null;
  applyView(E.override || E.mapPv);
}

// ---------------------------------------------------------------- toolbar / UI sync
/** Taxonomy / Rules tabs: the image is Inspect-only (no painting there). */
export function setInspectOnly(on) {
  if (on && !E.locked) { E.saveTool = E.tool; E.locked = true; setTool("inspect", true); }
  else if (!on && E.locked) { E.locked = false; setTool(E.saveTool, true); }
}
function setTool(t, force = false) {
  if (E.locked && !force) return;
  if (t === "sam" && !samAvailable()) { toast("SAM is not available", "error", 2000); return; }
  if (E.tool === "polygon" && t !== "polygon") E.poly = null;
  if (E.tool === "sam" && t !== "sam") sam.reset();
  E.tool = t; canvas.style.cursor = ""; syncUI(); syncStatus(); redraw();
}
function setSize(n) { E.size = clamp(Math.round(n), 1, 64); $("#eSize").value = E.size; $("#eSizeV").textContent = E.size; syncStatus(); redraw(); }
function bindToolbar() {
  $$("#eTools button").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
  $("#eSamNeg").addEventListener("click", () => { E.samNeg = !E.samNeg; syncUI(); });
  $("#eSize").addEventListener("input", (e) => setSize(Number(e.target.value)));
  $("#eShape").addEventListener("change", (e) => { E.round = e.target.value === "round"; e.target.blur(); syncUI(); redraw(); });
  $("#eOpacity").addEventListener("input", (e) => { E.opacity = Number(e.target.value) / 100; redraw(); });
  $("#eMask").addEventListener("click", () => { E.showMask = !E.showMask; syncUI(); redraw(); });
  $("#eOutline").addEventListener("click", () => { E.outline = !E.outline; syncUI(); paintRect(0, 0, E.W, E.H); });
  $("#eGrid").addEventListener("click", () => { E.showGrid = !E.showGrid; syncUI(); redraw(); });
  $("#eMap").addEventListener("change", (e) => setViewMap(e.target.value));
  $("#eFit").addEventListener("click", fit);
  $("#e11").addEventListener("click", () => zoomAt(E.cw / 2, E.ch / 2, 1));
  $("#eUndo").addEventListener("click", () => undoRedo(false));
  $("#eRedo").addEventListener("click", () => undoRedo(true));
  $("#eSave").addEventListener("click", () => save());
  $("#eRevert").addEventListener("click", revert);
  $$("#eStatus button").forEach((b) => b.addEventListener("click", () => setStatus(b.dataset.st)));
  $("#ePrev").addEventListener("click", () => step(-1));
  $("#eNext").addEventListener("click", () => step(1));
  $("#eNote").addEventListener("change", (e) => { if (!E.meta) return; E.meta.note = e.target.value; api(`/api/frames/${encodeURIComponent(E.stem)}/state`, { method: "PUT", body: { note: E.meta.note } }).catch(() => {}); });
  initHandle();
  $("#pane-edit").addEventListener("click", (e) => { if (e.target.closest("button")) e.target.closest("button").blur(); });
}
function syncUI() {
  $$("#eTools button").forEach((b) => { b.classList.toggle("on", b.dataset.tool === E.tool); if (b.dataset.tool === "sam") { b.disabled = !samAvailable(); b.title = samAvailable() ? "SAM2 (S)" : "SAM unavailable"; } });
  $("#eSamNeg").classList.toggle("on", E.samNeg);
  $("#eShape").value = E.round ? "round" : "square";
  $("#page-edit").dataset.dirty = String(history.dirty);
  $("#eMask").classList.toggle("on", E.showMask); $("#eOutline").classList.toggle("on", E.outline); $("#eGrid").classList.toggle("on", E.showGrid);
  $("#eUndo").disabled = !history.canUndo; $("#eRedo").disabled = !history.canRedo;
  $("#eSave").disabled = !history.dirty;
  $$("#eStatus button").forEach((b) => b.classList.toggle("on", b.dataset.st === E.meta?.status));
  const m = E.meta;
  const info = $("#eInfo");
  $("#eSamInfo").textContent = E.tool === "sam" ? sam.info : "";
  if (!m) { info.replaceChildren(); return; }
  info.replaceChildren(
    el("b", { class: "mono" }, E.stem),
    el("span", { class: `badge ${m.status}` }, m.status),
    el("span", { class: "muted" }, `${m.clip ?? ""} ${m.t_sec != null ? Number(m.t_sec).toFixed(1) + " s" : ""}`),
    el("span", { class: "muted" }, `conf ${m.conf_mean != null ? Number(m.conf_mean).toFixed(2) : "-"}  low ${m.lowconf_frac != null ? Number(m.lowconf_frac).toFixed(2) : "-"}`),
    el("span", { class: "muted" }, m.index != null && m.index >= 0 ? `${m.index + 1} / ${m.total}` : ""),
    el("span", { class: "muted" }, `edits ${m.edit_count ?? 0}`),
    history.dirty ? el("span", { class: "dirty" }, "unsaved") : "");
  if (document.activeElement !== $("#eNote")) $("#eNote").value = m.note || "";
  syncToolStatus();
}
function syncStatus() {
  status.set("zoom", `${E.z.toFixed(E.z < 10 ? 2 : 1)}x`);
  syncToolStatus();
}
const syncToolStatus = () => status.set("tool", E.tool === "inspect" && E.sel ? `Inspect: ${selText()}` : E.tool === "sam" && sam.info ? `SAM ${sam.info}` : `${TOOL_NAMES[E.tool]}${E.tool === "brush" || E.tool === "erase" ? " " + E.size + (E.round ? "r" : "s") : ""}${picker?.protect >= 0 ? " | protect " + clsName(picker.protect) : ""}`);

// ---------------------------------------------------------------- keyboard
export function onKeyUp(e) { if (e.key === " " && E.space) { E.space = false; canvas.classList.remove("pan"); redraw(); return true; } return false; }
export function onKey(e) {
  const k = e.key, mod = e.ctrlKey || e.metaKey, lk = k.toLowerCase();
  if (mod) {
    if (lk === "z") { undoRedo(e.shiftKey); return true; }
    if (lk === "y") { undoRedo(true); return true; }
    if (lk === "s") { save(); return true; }
    return false;
  }
  if (e.altKey) return false;
  if (/^[1-9]$/.test(k)) { picker.byDigit(Number(k)); return true; }
  switch (lk) {
    case " ": if (!E.space) { E.space = true; canvas.classList.add("pan"); redraw(); } return true;
    case "v": setTool("inspect"); return true;
    case "b": setTool("brush"); return true;
    case "e": setTool("erase"); return true;
    case "r": setTool("region"); return true;
    case "l": setTool("polygon"); return true;
    case "s": setTool("sam"); return true;
    case "[": setSize(E.size - (E.size > 24 ? 4 : E.size > 8 ? 2 : 1)); return true;
    case "]": setSize(E.size + (E.size >= 24 ? 4 : E.size >= 8 ? 2 : 1)); return true;
    case "q": E.round = !E.round; syncUI(); redraw(); return true;
    case "f": fit(); return true;
    case "0": E.cw && zoomAt(E.cw / 2, E.ch / 2, 1); return true;
    case "m": E.showMask = !E.showMask; syncUI(); redraw(); return true;
    case "o": E.outline = !E.outline; syncUI(); paintRect(0, 0, E.W, E.H); return true;
    case "g": E.showGrid = !E.showGrid; syncUI(); redraw(); return true;
    case "x": picker.swap(); return true;
    case "=": case "+": E.cw && zoomAt(E.cw / 2, E.ch / 2, E.z * 1.5); return true;
    case "-": E.cw && zoomAt(E.cw / 2, E.ch / 2, E.z / 1.5); return true;
    case "n": case "arrowright": step(1); return true;
    case "p": case "arrowleft": step(-1); return true;
    case "/": picker.focusSearch(); return true;
    case "tab": if (E.tool === "sam") { sam.cycle(e.shiftKey ? -1 : 1); return true; } return false;
    case "enter": if (E.tool === "inspect" && E.sel && !E.locked) { relabelSel(); return true; } if (E.tool === "polygon" && E.poly) { closePoly(); return true; } if (E.tool === "sam" && sam.has) { applySam(); return true; } return false;
    case "escape": if (E.sel) { clearSel(); return true; } if (E.poly) { E.poly = null; redraw(); return true; } if (E.tool === "sam") { sam.reset(); return true; } return false;
  }
  return false;
}
/** Previous / next frame of the filter; wraps (next on the last -> first, prev on the first -> last). */
export async function step(dir) {
  if (!E.meta) return;
  let t = dir < 0 ? E.meta.prev : E.meta.next;
  if (!t) {
    const total = E.meta.total;
    if (!total) { toast("No frames in this filter", "ok", 1500); return; }
    t = await edgeStem(dir < 0 ? total - 1 : 0);
    if (!t || t === E.stem) return;                       // a single-frame filter stays on its frame
  }
  go(t);
}
async function edgeStem(offset) {
  const q = new URLSearchParams(E.qs); q.set("offset", offset); q.set("limit", 1);
  try { return (await api(`/api/frames?${q}`)).rows[0]?.stem ?? ""; } catch { return ""; }
}
/** The browse filter changed: refresh prev/next/index for the current frame without reloading it. */
export async function setFilter(qs) {
  E.qs = qs;
  if (!E.stem) return;
  try {
    const m = await api(`/api/frames/${encodeURIComponent(E.stem)}${qs ? "?" + qs : ""}`, { quiet: true });
    Object.assign(E.meta, { prev: m.prev, next: m.next, index: m.index, total: m.total });
    syncUI();
    document.dispatchEvent(new CustomEvent("frame-opened", { detail: { stem: E.stem, index: m.index, total: m.total } }));
    if (m.index == null || m.index < 0) await goFirst();
  } catch { Object.assign(E.meta, { prev: null, next: null, index: -1 }); syncUI(); await goFirst(); }
}
async function goFirst() { const t = await edgeStem(0); if (t && t !== E.stem) go(t); }   // the current frame left the filter
function initHandle() {
  const stage = $("#eStage"), h = $("#eHandle");
  let saved = 0;
  try { saved = Number(localStorage.getItem("lab.imgH")) || 0; } catch { /* ignore */ }
  const setH = (v, persist) => {
    v = clamp(Math.round(v), 120, Math.max(300, window.innerHeight * 0.96)); stage.style.height = v + "px";
    if (persist) { try { localStorage.setItem("lab.imgH", String(v)); } catch { /* ignore */ } }
  };
  const def = () => (window.innerWidth < 600 ? Math.round((window.innerWidth - 22) * 9 / 16) + 2 : clamp(window.innerHeight - 400, 320, 760));
  setH(saved || def(), false);
  h.addEventListener("pointerdown", (e) => {
    e.preventDefault(); h.setPointerCapture(e.pointerId); h.classList.add("drag");
    const y0 = e.clientY, h0 = stage.getBoundingClientRect().height;
    const mv = (ev) => setH(h0 + ev.clientY - y0, false);
    const up = (ev) => { h.classList.remove("drag"); h.removeEventListener("pointermove", mv); h.removeEventListener("pointerup", up); setH(h0 + ev.clientY - y0, true); };
    h.addEventListener("pointermove", mv); h.addEventListener("pointerup", up);
  });
  h.addEventListener("dblclick", () => { try { localStorage.removeItem("lab.imgH"); } catch { /* ignore */ } setH(def(), false); });
}
export const refreshMeta = async () => { await loadMeta(); syncUI(); };
