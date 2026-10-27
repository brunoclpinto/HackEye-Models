// Small shared helpers: DOM, toast, debounce, modal dialogs, status bar, job polling.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Tiny element builder: el("div", {class:"x", onclick:fn, dataset:{a:1}}, child, "text"). */
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else if (k === "dataset") Object.assign(n.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k in n && k !== "list") { try { n[k] = v; } catch { n.setAttribute(k, v); } }
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
}

export function toast(msg, kind = "error", ms = 6000) {
  const host = $("#toasts");
  if (!host) { console.error(msg); return; }
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  t.addEventListener("click", () => t.remove());
  host.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export function debounce(fn, ms = 200) {
  let h = 0;
  const d = (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); };
  d.flush = (...a) => { clearTimeout(h); fn(...a); };
  d.cancel = () => clearTimeout(h);
  return d;
}

export const typing = (t) => !!t && ((t.tagName === "INPUT" && !["checkbox", "range", "button"].includes(t.type)) ||
  t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);

export const rgbCss = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
export const rgbHex = (c) => "#" + c.map((v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0")).join("");
export const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";
export const fmtPx = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n));

/** Status bar cells: status.set("tool", "Brush"). */
export const status = {
  set(key, text) { const e = $("#sb-" + key); if (e && e.textContent !== text) e.textContent = text; },
};

/**
 * Modal dialog. build(body) fills the body and returns a getter `() => value | undefined`
 * (undefined = invalid, keep open). Resolves with the value, or null when cancelled.
 */
export function dialog(title, build, okText = "OK") {
  return new Promise((resolve) => {
    const host = $("#modal");
    const body = el("div", { class: "dlg-body" });
    const get = build(body);
    const prevFocus = document.activeElement;
    const done = (v) => { host.hidden = true; host.replaceChildren(); window.removeEventListener("keydown", onKey, true); prevFocus?.focus?.(); resolve(v); };
    const ok = () => { const v = get(); if (v !== undefined) done(v); };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); }
      else if (e.key === "Enter" && e.target.tagName !== "TEXTAREA" && e.target.tagName !== "BUTTON" && e.target.tagName !== "SELECT") { e.preventDefault(); e.stopPropagation(); ok(); }
    };
    window.addEventListener("keydown", onKey, true);
    const box = el("div", { class: "card dlg" },
      el("h2", {}, title), body,
      el("div", { class: "row end" }, el("button", { onclick: () => done(null) }, "Cancel"), el("button", { class: "primary", "data-testid": "dlg-ok", onclick: ok }, okText)));
    host.replaceChildren(box);
    host.hidden = false;
    host.onmousedown = (e) => { if (e.target === host) done(null); };
    setTimeout(() => (body.querySelector("input,select,textarea,button") || box).focus(), 0);
  });
}

export const confirmDialog = (msg, okText = "OK") =>
  dialog("Confirm", (b) => { b.append(el("p", {}, msg)); return () => true; }, okText).then((v) => v === true);

/** Poll GET /api/jobs/{id} until done/failed. onUpdate(job) each tick. */
export async function pollJob(api, id, onUpdate, ms = 800) {
  for (;;) {
    let job;
    try { job = await api(`/api/jobs/${id}`, { quiet: true }); }
    catch (e) { onUpdate({ status: "failed", progress: 0, log: String(e.message) }); throw e; }
    onUpdate(job);
    if (job.status !== "running") return job;
    await new Promise((r) => setTimeout(r, ms));
  }
}

/** Render a job into a container (progress bar + log tail + result). */
export function renderJob(host, job, label = "") {
  host.hidden = false;
  host.dataset.status = job.status;
  const p = Math.round(clamp(job.progress || 0, 0, 1) * 100);
  let result = job.result;
  if (typeof result === "string") { try { result = JSON.parse(result); } catch { /* keep string */ } }
  host.replaceChildren(
    el("div", { class: "row" }, el("b", {}, `${label} ${job.status}`), el("span", { class: "muted mono" }, `${p}%`)),
    el("div", { class: "bar-outer" }, el("div", { class: `bar-inner ${job.status}`, style: { width: p + "%" } })),
    job.log ? el("pre", { class: "log" }, String(job.log).split("\n").slice(-8).join("\n")) : null,
    result ? el("pre", { class: "log res" }, typeof result === "string" ? result : JSON.stringify(result, null, 1)) : null);
}
