// Browse group: filter bar (bound to the URL hash) + horizontal virtual-scrolled thumbnail carousel.
import { api } from "./api.js";
import { $, el, debounce, clamp } from "./util.js";
import { S, on } from "./store.js";
import { nav, hashFor, replaceHash } from "./nav.js";
import * as editor from "./editor.js";

const KEYS = ["split", "clip", "status", "cmin", "cmax", "lmin", "lmax", "cls", "minpx", "edited", "sort"];
const PAGE = 200, CW = 176;
const B = { qs: null, rows: [], total: 0, loading: new Set(), gen: 0, cells: new Map(), clips: [], curStem: "", drag: null, moved: false };
let form, scroll, spacer, loadP = Promise.resolve();

export function init() {
  form = $("#bFilter"); scroll = $("#carousel"); spacer = $("#cSpacer");
  form.addEventListener("input", debounce(() => applyForm(), 250));
  form.addEventListener("change", (e) => { if (e.target.name === "split") fillClips(); applyForm(); });
  $("#bReset").addEventListener("click", () => { location.hash = hashFor(nav.stem, nav.tab, ""); });
  scroll.addEventListener("scroll", render, { passive: true });
  new ResizeObserver(() => render()).observe(scroll);
  scroll.addEventListener("wheel", (e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.preventDefault(); scroll.scrollLeft += e.deltaY * (e.deltaMode === 1 ? 16 : 1); } }, { passive: false });
  scroll.addEventListener("pointerdown", dragStart); scroll.addEventListener("pointermove", dragMove);
  scroll.addEventListener("pointerup", dragEnd); scroll.addEventListener("pointercancel", dragEnd);
  $("#bTags").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { editor.setStatus(b.dataset.st); b.blur(); } });
  $("#bPrev").addEventListener("click", (e) => { editor.step(-1); e.currentTarget.blur(); });
  $("#bNext").addEventListener("click", (e) => { editor.step(1); e.currentTarget.blur(); });
  $("#bFirst").addEventListener("click", (e) => { jump("first"); e.currentTarget.blur(); });
  $("#bLast").addEventListener("click", (e) => { jump("last"); e.currentTarget.blur(); });
  document.addEventListener("frame-updated", (e) => patchRow(e.detail));
  document.addEventListener("frame-opened", (e) => { B.curStem = e.detail.stem; mark(); centerOn(e.detail.index); });
  document.addEventListener("frames-changed", () => { if (B.qs !== null) load(B.qs); });
  on("taxonomy", fillClasses);
  fillClasses();
  api("/api/clips", { quiet: true }).then((c) => { B.clips = c; fillClips(); if (B.qs !== null) writeForm(B.qs); }).catch(() => {});
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
  for (const k of KEYS) { const e = form.elements[k]; if (e) e.value = q.get(k) ?? (k === "sort" ? "time" : ""); }
  fillClips();
  const clip = q.get("clip"); if (clip) form.elements.clip.value = clip;
}
function applyForm() {
  const qs = readForm();
  if (qs === B.qs) return;
  nav.qs = qs; replaceHash(hashFor(nav.stem, nav.tab, qs));
  load(qs); editor.setFilter(qs);
}

/** Router entry: make the carousel show the list for `qs`. Resolves once the first page is in. */
export function setFilter(qs) {
  writeForm(qs);
  if (qs === B.qs) return loadP;
  return load(qs);
}
export const firstStem = () => B.rows[0]?.stem ?? "";

function load(qs) {
  B.qs = qs; B.gen++; B.rows = []; B.total = 0; B.loading.clear(); clearCells();
  scroll.scrollLeft = 0;
  $("#bCount").textContent = "loading...";
  loadP = fetchPage(0).then(() => { render(); if (B.curStem) centerOn(B.rows.findIndex((r) => r && r.stem === B.curStem)); });
  return loadP;
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
    spacer.style.width = r.total * CW + "px";
    for (const [i, c] of [...B.cells]) if (((i / PAGE) | 0) === p) { c.remove(); B.cells.delete(i); }
    render();
  } catch { B.loading.delete(p); }
}
function clearCells() { for (const c of B.cells.values()) c.remove(); B.cells.clear(); }

function render() {
  const w = scroll.clientWidth;
  if (!w || !B.total) return;
  const i0 = Math.max(0, Math.floor(scroll.scrollLeft / CW) - 3), i1 = Math.min(B.total, Math.ceil((scroll.scrollLeft + w) / CW) + 3);
  for (const [i, c] of [...B.cells]) if (i < i0 || i >= i1) { c.remove(); B.cells.delete(i); }
  for (let i = i0; i < i1; i++) {
    const row = B.rows[i];
    if (!row) { const p = (i / PAGE) | 0; if (!B.loading.has(p)) fetchPage(p); continue; }
    if (B.cells.has(i)) continue;
    const c = makeCell(i, row);
    c.style.left = i * CW + "px"; c.style.width = CW + "px";
    B.cells.set(i, c); spacer.append(c);
  }
}
function makeCell(i, row) {
  const c = el("div", { class: `cell${row.stem === B.curStem ? " cur" : ""}`, "data-testid": "frame-card", dataset: { i, stem: row.stem, split: row.split, status: row.status } },
    el("div", { class: "in" },
      el("img", { src: `/api/frames/${encodeURIComponent(row.stem)}/thumb.jpg`, loading: "lazy", decoding: "async", draggable: false }),
      el("div", { class: "meta" },
        el("span", { class: "stem mono", title: row.stem }, shortStem(row.stem)),
        el("span", { class: "muted mono" }, `${Number(row.conf_mean).toFixed(2)}${row.edit_count ? " e" + row.edit_count : ""}`),
        el("span", { class: `badge ${row.status}` }, row.status))));
  c.addEventListener("click", () => { if (!B.moved) location.hash = hashFor(row.stem); });
  return c;
}
// "Chest-center-0001_000215" -> "C-0001 · 000215" (the full stem stays in the tooltip)
function shortStem(stem) {
  const m = /^([A-Za-z])[A-Za-z]*-[A-Za-z]+-(\d+)_(\d+)$/.exec(stem);
  return m ? `${m[1]}-${m[2]} · ${m[3]}` : stem;
}
function mark() { for (const c of B.cells.values()) c.classList.toggle("cur", c.dataset.stem === B.curStem); }
function centerOn(i) {
  if (i == null || i < 0 || !scroll.clientWidth) return;
  scroll.scrollTo({ left: Math.max(0, i * CW - (scroll.clientWidth - CW) / 2), behavior: "smooth" });
}

function dragStart(e) { if (e.button !== 0) return; B.drag = { x: e.clientX, left: scroll.scrollLeft, id: e.pointerId }; B.moved = false; }
function dragMove(e) {
  const d = B.drag; if (!d) return;
  const dx = e.clientX - d.x;
  if (!B.moved && Math.abs(dx) > 5) { B.moved = true; scroll.classList.add("dragging"); try { scroll.setPointerCapture(d.id); } catch { /* ignore */ } }
  if (B.moved) scroll.scrollLeft = d.left - dx;
}
function dragEnd() { B.drag = null; scroll.classList.remove("dragging"); setTimeout(() => { B.moved = false; }, 0); }

function patchRow({ stem, status, edit_count }) {
  const i = B.rows.findIndex((r) => r && r.stem === stem);
  if (i < 0) return;
  if (status) B.rows[i].status = status;
  if (edit_count != null) B.rows[i].edit_count = edit_count;
  const c = B.cells.get(i); if (c) { c.remove(); B.cells.delete(i); }
  render();
}

/** Home / End: open the first or last frame of the filter. */
export async function jump(which) {
  if (!B.total) return;
  let stem;
  if (which === "first") stem = B.rows[0]?.stem;
  else {
    const q = new URLSearchParams(B.qs); q.set("offset", B.total - 1); q.set("limit", 1);
    try { stem = (await api(`/api/frames?${q}`)).rows[0]?.stem; } catch { return; }
  }
  if (stem) location.hash = hashFor(stem);
}
