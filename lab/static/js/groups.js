// Collapsible groups. State persists in localStorage (per group id) inside try/catch; the page works without it.
import { el } from "./util.js";

const key = (id) => "lab.grp." + id;
export function stored(id) {
  try { const v = localStorage.getItem(key(id)); return v === null ? null : v === "1"; } catch { return null; }
}
function store(id, c) { try { localStorage.setItem(key(id), c ? "1" : "0"); } catch { /* ignore */ } }

export function setCollapsed(root, c, persist = false) {
  root.dataset.collapsed = String(c);
  const t = root.querySelector(":scope > .gh .gt"); if (t) t.setAttribute("aria-expanded", String(!c));
  if (persist) store(root.dataset.gid, c);
  root.dispatchEvent(new CustomEvent("group-toggle", { bubbles: false }));
}
/** Apply `c` only when the user has not chosen a state for this group. */
export function applyDefault(root, c) { if (stored(root.dataset.gid) === null) setCollapsed(root, c); }

/** Wire a group whose markup is `.grp[data-gid] > .gh > button.gt` + `.gb`. */
export function bindGroup(root, defCollapsed = false) {
  const id = root.dataset.gid;
  root.setAttribute("data-testid", `group-${id}`);
  const t = root.querySelector(":scope > .gh .gt");
  t.setAttribute("data-testid", "group-toggle"); t.type = "button";
  const s = stored(id);
  setCollapsed(root, s === null ? defCollapsed : s);
  t.addEventListener("click", () => { setCollapsed(root, root.dataset.collapsed !== "true", true); t.blur(); });
  return root;
}
export function bindAll(host = document) { host.querySelectorAll(".grp[data-gid]").forEach((g) => { if (!g.hasAttribute("data-testid")) bindGroup(g, g.dataset.def === "collapsed"); }); }

/** Build a group element. Returns {root, body, head}. */
export function makeGroup(id, title, { sub = true, extra = null, collapsed = false, count = "" } = {}) {
  const body = el("div", { class: "gb" });
  const head = el("header", { class: "gh" }, el("button", { class: "gt" }, el("i", { class: "chev" }), el("span", {}, title), count ? el("small", { class: "muted" }, count) : null), extra);
  const root = el("section", { class: `grp${sub ? " sub" : ""}`, dataset: { gid: id } }, head, body);
  bindGroup(root, collapsed);
  return { root, body, head };
}
