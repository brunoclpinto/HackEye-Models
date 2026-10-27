// Browse tab: filter bar bound to the URL hash + virtual-scrolled thumbnail grid.
import { api } from "./api.js";
import { $, el, debounce, toast, clamp } from "./util.js";
import { S, on } from "./store.js";

const KEYS = ["split", "clip", "status", "cmin", "cmax", "lmin", "lmax", "cls", "minpx", "edited", "sort"];
const PAGE = 200, MIN_CELL = 236, STATUS_KEY = { 1: "ok", 2: "fix", 3: "drop", 4: "todo" };
const B = { qs: "", rows: [], total: 0, loading: new Set(), gen: 0, cells: new Map(), cols: 1, rowH: 160, sel: new Set(), anchor: 0, focus: 0, pendingFocus: null, clips: [] };
let form, scroll, spacer, ready = false;

export const currentQS = () => B.qs;

export function init() {
  form = $("#bFilter"); scroll = $("#bScroll"); spacer = $("#bSpacer");
  form.addEventListener("input", debounce(() => applyForm(), 250));
  form.addEventListener("change", (e) => { if (e.target.name === "split") fillClips(); applyForm(); });
  $("#bReset").addEventListener("click", () => { location.hash = "#browse"; });
  scroll.addEventListener("scroll", () => render(), { passive: true });
  new ResizeObserver(() => { layout(); render(); }).observe(scroll);
  document.addEventListener("frame-updated", (e) => patchRow(e.detail));
  document.addEventListener("frame-opened", (e) => { B.pendingFocus = e.detail; });
  document.addEventListener("frames-changed", () => { if (ready) load(B.qs); });
  on("taxonomy", fillClasses);
  fillClasses();
  api("/api/clips", { quiet: true }).then((c) => { B.clips = c; fillClips(); }).catch(() => {});
}

function fillClasses() {
  const s = form.elements.cls, cur = s.value;
  s.replaceChildren(el("option", { value: "" }, "any"), ...[...S.tax].sort((a, b) => a.name.localeCompare(b.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`)));
  s.value = cur;
}
function fillClips() {
  const s = form.elements.clip, cur = s.value, split = form.elements.split.value;
  s.replaceChildren(el("option", { value: "" }, "all"), ...B.clips.filter((c) => !split || c.split === split).map((c) => el("option", { value: c.clip }, `${c.clip} (${c.frames})`)));
  s.value = [...s.options].some((o) => o.value === cur) ? cur : "";
}

function readForm() {
  const q = new URLSearchParams();
  for (const k of KEYS) { const v = form.elements[k].value.trim(); if (v !== "" && !(k === "sort" && v === "time")) q.set(k, v); }
  return q.toString();
}
function writeForm(qs) {
  const q = new URLSearchParams(qs);
  for (const k of KEYS) { const e = form.elements[k]; if (!e) continue; e.value = q.get(k) ?? (k === "sort" ? "time" : ""); }
  fillClips();
  const clip = q.get("clip"); if (clip) form.elements.clip.value = clip;
}
function applyForm() {
  const qs = readForm();
  if (qs === B.qs) return;
  history.replaceState(null, "", `#browse${qs ? "?" + qs : ""}`);
  load(qs);
}

/** Called by the router when the Browse tab is shown. */
export function show(qs) {
  writeForm(qs);
  requestAnimationFrame(() => { layout(); if (!ready || qs !== B.qs) load(qs); else { focusPending(); render(); } });
}

async function load(qs) {
  ready = true;
  B.qs = qs; B.gen++; B.rows = []; B.total = 0; B.loading.clear(); clearCells(); B.sel.clear(); B.focus = 0; B.anchor = 0;
  scroll.scrollTop = 0;
  $("#bCount").textContent = "loading...";
  await fetchPage(0);
  layout(); if (B.total && !B.pendingFocus) B.sel = new Set([0]); focusPending(); render();
}
async function fetchPage(p) {
  if (B.loading.has(p)) return;
  B.loading.add(p);
  const gen = B.gen, q = new URLSearchParams(B.qs);
  q.set("offset", p * PAGE); q.set("limit", PAGE);
  try {
    const r = await api(`/api/frames?${q}`);
    if (gen !== B.gen) return;
    B.total = r.total;
    r.rows.forEach((row, i) => { B.rows[p * PAGE + i] = row; });
    $("#bCount").textContent = `${r.total} frames`;
    for (const [i, c] of [...B.cells]) if (((i / PAGE) | 0) === p) { c.remove(); B.cells.delete(i); }
    layout(); render();
  } catch { B.loading.delete(p); }
}
function clearCells() { for (const c of B.cells.values()) c.remove(); B.cells.clear(); }

function layout() {
  const w = scroll.clientWidth;
  if (!w) return;
  const cols = Math.max(1, Math.floor(w / MIN_CELL)), cw = w / cols;
  if (cols !== B.cols) clearCells();
  B.cols = cols; B.cw = cw; B.rowH = Math.round((cw - 12) * 9 / 16 + 36);
  spacer.style.height = Math.ceil(B.total / cols) * B.rowH + "px";
}
function focusPending() {
  const f = B.pendingFocus; B.pendingFocus = null;
  if (!f || f.index == null || f.index < 0 || f.index >= Math.max(B.total, 1)) return;
  B.focus = B.anchor = f.index; B.sel = new Set([f.index]);
  ensureVisible(f.index, true);
}
function ensureVisible(i, center = false) {
  const top = Math.floor(i / B.cols) * B.rowH, h = scroll.clientHeight;
  if (center) scroll.scrollTop = Math.max(0, top - (h - B.rowH) / 2);
  else if (top < scroll.scrollTop) scroll.scrollTop = top;
  else if (top + B.rowH > scroll.scrollTop + h) scroll.scrollTop = top + B.rowH - h;
}

function render() {
  if (!B.cw) return;
  const h = scroll.clientHeight, r0 = Math.max(0, Math.floor(scroll.scrollTop / B.rowH) - 1), r1 = Math.ceil((scroll.scrollTop + h) / B.rowH) + 1;
  const i0 = r0 * B.cols, i1 = Math.min(B.total, (r1 + 1) * B.cols);
  for (const [i, c] of [...B.cells]) if (i < i0 || i >= i1) { c.remove(); B.cells.delete(i); }
  for (let i = i0; i < i1; i++) {
    const row = B.rows[i];
    if (!row) { const p = (i / PAGE) | 0; if (!B.loading.has(p)) fetchPage(p); continue; }
    let c = B.cells.get(i);
    if (!c) { c = makeCell(i, row); B.cells.set(i, c); spacer.append(c); }
    c.style.left = (i % B.cols) * B.cw + "px"; c.style.top = Math.floor(i / B.cols) * B.rowH + "px";
    c.style.width = B.cw + "px"; c.style.height = B.rowH + "px";
    c.classList.toggle("sel", B.sel.has(i));
  }
}
function makeCell(i, row) {
  const c = el("div", { class: "cell", "data-testid": "frame-card", dataset: { i, stem: row.stem, split: row.split, status: row.status } },
    el("div", { class: "in" },
      el("img", { src: `/api/frames/${encodeURIComponent(row.stem)}/thumb.jpg`, loading: "lazy", decoding: "async", draggable: false }),
      el("div", { class: "meta" },
        el("span", { class: "stem mono", title: row.stem }, row.stem),
        el("span", { class: "muted mono" }, `${Number(row.conf_mean).toFixed(2)}${row.edit_count ? " e" + row.edit_count : ""}`),
        el("span", { class: `badge ${row.status}` }, row.status))));
  c.addEventListener("click", (e) => select(i, e));
  c.addEventListener("dblclick", () => open(i));
  return c;
}
function select(i, e) {
  if (e.shiftKey) { const a = Math.min(B.anchor, i), b = Math.max(B.anchor, i); B.sel = new Set(); for (let k = a; k <= b; k++) B.sel.add(k); }
  else if (e.metaKey || e.ctrlKey) { B.sel.has(i) ? B.sel.delete(i) : B.sel.add(i); B.anchor = i; }
  else { B.sel = new Set([i]); B.anchor = i; }
  B.focus = i; scroll.focus({ preventScroll: true }); render();
}
function open(i) { const r = B.rows[i]; if (r) location.hash = `#edit/${r.stem}${B.qs ? "?" + B.qs : ""}`; }

function patchRow({ stem, status, edit_count }) {
  const i = B.rows.findIndex((r) => r && r.stem === stem);
  if (i < 0) return;
  if (status) B.rows[i].status = status;
  if (edit_count != null) B.rows[i].edit_count = edit_count;
  const c = B.cells.get(i); if (c) { c.remove(); B.cells.delete(i); }
  if (ready) render();
}

async function tag(status) {
  const idx = [...B.sel].filter((i) => B.rows[i]);
  if (!idx.length) return;
  await Promise.all(idx.map((i) => api(`/api/frames/${encodeURIComponent(B.rows[i].stem)}/state`, { method: "PUT", body: { status } }).then(() => patchRow({ stem: B.rows[i].stem, status })).catch(() => {})));
}

function onKey(e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const move = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: B.cols, ArrowUp: -B.cols }[e.key];
  if (move !== undefined) { e.preventDefault(); moveFocus(move, e.shiftKey); }
}
function moveFocus(d, extend) {
  if (!B.total) return;
  const i = clamp(B.focus + d, 0, B.total - 1);
  if (extend) { const a = Math.min(B.anchor, i), b = Math.max(B.anchor, i); B.sel = new Set(); for (let k = a; k <= b; k++) B.sel.add(k); }
  else { B.sel = new Set([i]); B.anchor = i; }
  B.focus = i; ensureVisible(i); render();
}

/** Router key hook (Browse tab active, not typing). Returns true if handled. */
export function onKeyGlobal(e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return false;
  if (STATUS_KEY[e.key]) { tag(STATUS_KEY[e.key]); return true; }
  if (e.key === "Enter") { open(B.focus); return true; }
  if (["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) { onKey(e); return true; }
  return false;
}
