// Browse group: filter bar (bound to the URL hash) + one navigation row (Prev, position, Next, tags, count).
// Navigation wraps (last -> first, first -> last); the wrapping itself lives in editor.step().
import { api } from "./api.js";
import { $, el, debounce } from "./util.js";
import { S, on } from "./store.js";
import { nav, hashFor, replaceHash } from "./nav.js";
import * as editor from "./editor.js";

const KEYS = ["split", "clip", "status", "cmin", "cmax", "lmin", "lmax", "cls", "minpx", "edited", "sort"];
const B = { qs: null, total: 0, first: "", clips: [], gen: 0 };
let form, loadP = Promise.resolve();

export function init() {
  form = $("#bFilter");
  form.addEventListener("input", debounce(() => applyForm(), 250));
  form.addEventListener("change", (e) => { if (e.target.name === "split") fillClips(); applyForm(); });
  $("#bReset").addEventListener("click", () => { location.hash = hashFor(nav.stem, nav.tab, ""); });
  $("#bTags").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { editor.setStatus(b.dataset.st); b.blur(); } });
  $("#bPrev").addEventListener("click", (e) => { editor.step(-1); e.currentTarget.blur(); });
  $("#bNext").addEventListener("click", (e) => { editor.step(1); e.currentTarget.blur(); });
  document.addEventListener("frame-opened", (e) => setPos(e.detail.index, e.detail.total));
  document.addEventListener("frames-changed", () => { if (B.qs !== null) load(B.qs); });
  on("taxonomy", fillClasses);
  fillClasses();
  api("/api/clips", { quiet: true }).then((c) => { B.clips = c; fillClips(); if (B.qs !== null) writeForm(B.qs); }).catch(() => {});
}

function setPos(index, total) {
  const p = $("#bPos");
  if (!B.total && !total) { p.textContent = "no frames"; p.dataset.index = "0"; p.dataset.total = "0"; return; }
  const t = total ?? B.total;
  p.textContent = `${index != null && index >= 0 ? index + 1 : "-"} / ${t}`;
  p.dataset.index = String(index != null && index >= 0 ? index + 1 : 0); p.dataset.total = String(t);
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
  load(qs).then(() => editor.setFilter(qs));
}

/** Router entry: make the count / first frame reflect `qs`. Resolves once known. */
export function setFilter(qs) {
  writeForm(qs);
  if (qs === B.qs) return loadP;
  return load(qs);
}
export const firstStem = () => B.first;

function load(qs) {
  B.qs = qs; const gen = ++B.gen;
  $("#bCount").textContent = "loading...";
  const q = new URLSearchParams(qs); q.set("offset", 0); q.set("limit", 1);
  loadP = api(`/api/frames?${q}`).then((r) => {
    if (gen !== B.gen) return;
    B.total = r.total; B.first = r.rows[0]?.stem ?? "";
    $("#bCount").textContent = `${r.total} frames`;
    if (!r.total) setPos(-1, 0);
  }).catch(() => { $("#bCount").textContent = ""; });
  return loadP;
}
