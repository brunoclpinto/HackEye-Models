// Class picker: search + selected class, then two side-by-side sub-groups: Detected (classes in the open frame, flat chips
// with their detection count) and Undetected (the rest, grouped by hier, collapsed). Recents (digits 1-9), secondary class
// and protect. No class is active until one is picked.
import { el, rgbCss, debounce } from "./util.js";
import { S, on } from "./store.js";
import { makeGroup, setCollapsed, stored } from "./groups.js";

export class Picker {
  constructor(host, protHost, onChange, extraHost = null, onDetection = null) {
    this.host = host; this.protHost = protHost; this.extraHost = extraHost; this.onChange = onChange; this.onDetection = onDetection;
    this.dets = null; this.detSel = null; this.detAll = false;
    this.active = null; this.secondary = 0; this.protect = -1;
    this.recents = [];
    try { this.recents = JSON.parse(localStorage.getItem("lab.recents") || "[]"); } catch { /* ignore */ }
    this.items = new Map(); this.groups = new Map(); this.counts = null;
    this.build();
    on("taxonomy", () => this.build());
  }
  build() {
    const tax = S.tax;
    if (!tax.length) return;
    if (this.active != null && !S.byId.has(this.active)) this.active = null;
    if (!S.byId.has(this.secondary)) this.secondary = tax[0].id;
    this.recents = this.recents.filter((id) => S.byId.has(id)).slice(0, 9);

    this.curBox = el("div", { class: "pk-cur", "data-testid": "active-class", hidden: true }, this.curSw = el("span", { class: "sw big" }), this.curName = el("b"),
      el("button", { class: "pk-x", type: "button", "data-testid": "active-class-clear", title: "Clear the selected class (Esc)", "aria-label": "Clear the selected class", onclick: () => this.setActive(null) }, "\u00d7"));
    this.secSw = el("span", { class: "sw" }); this.secName = el("span", { class: "muted" });
    this.search = el("input", { "data-testid": "picker-search", type: "search", placeholder: "Search classes (/)", autocomplete: "off" });
    this.search.addEventListener("input", debounce(() => this.filter(), 60));
    this.search.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { const first = [...this.items.values()].find((n) => !n.hidden); if (first) { this.pick(Number(first.dataset.id)); this.search.blur(); } }
      else if (e.key === "Escape") { if (!this.search.value) this.setActive(null); else { this.search.value = ""; this.filter(); } this.search.blur(); }
    });
    this.protSel = el("select", { "data-testid": "protect-select", title: "Protect source class: tools only change pixels of this class (Shift+Alt+click on the image)" },
      el("option", { value: "" }, "none"),
      ...[...tax].sort((a, b) => a.name.localeCompare(b.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`)));
    this.protSel.addEventListener("change", () => { this.protect = this.protSel.value === "" ? -1 : Number(this.protSel.value); this.protSel.blur(); this.onChange?.(); });
    this.recentsEl = el("div", { class: "pk-recents" });

    // Detected: flat list; Undetected: one collapsible group per hier. The chips are built once and moved between them.
    this.det = makeGroup("edit-detected", "Detected", { count: "0" });
    this.undet = makeGroup("edit-undetected", "Undetected", { count: "0" });
    this.det.body.classList.add("pk-flat"); this.undet.body.classList.add("pk-col");
    this.items.clear(); this.groups.clear();
    const hiers = new Map();
    for (const c of tax) { if (!hiers.has(c.hier)) hiers.set(c.hier, []); hiers.get(c.hier).push(c); }
    for (const [hier, cs] of hiers) {
      const g = makeGroup(`hier-${hier || "none"}`, hier || "(none)", { count: `${cs.length}`, collapsed: true });
      g.root.dataset.hier = hier; g.root._body = g.body; g.root._ids = cs.map((c) => c.id); g.root._cnt = g.root.querySelector(".gh small");
      this.groups.set(hier, g.root); this.undet.body.append(g.root);
    }
    for (const c of tax) {
      const it = el("span", { class: "pk-item", "data-testid": "picker-item", dataset: { id: c.id } },
        el("span", { class: "sw", style: { background: rgbCss(c.color) } }), el("span", { class: "nm" }, c.name), el("span", { class: "muted mono cnt", hidden: true }));
      it.dataset.q = `${c.name} ${c.raw_name} ${c.id}`.toLowerCase();
      it._cnt = it.querySelector(".cnt"); it._hier = c.hier; it._c = c;
      it.addEventListener("click", () => { if (this.active === c.id) this.setActive(null); else this.pick(c.id); });
      it.addEventListener("contextmenu", (e) => { e.preventDefault(); this.setSecondary(c.id); });
      it.addEventListener("dblclick", () => this.setProtect(c.id));
      this.items.set(c.id, it);
    }
    this.host.replaceChildren(
      el("div", { class: "pk-head" }, this.search, this.curBox),
      this.detEl = el("div", { class: "pk-dets", "data-testid": "detections", hidden: true }),
      el("div", { class: "pk-lists" }, this.det.root, this.undet.root));
    this.protHost.replaceChildren(el("label", { class: "muted" }, "Only change pixels of ", this.protSel));
    this.extraHost?.replaceChildren(
      el("span", { class: "pk-cur", title: "Secondary class for the Erase tool (right-click a chip, X swaps)" }, el("span", { class: "muted" }, "Erase to"), this.secSw, this.secName),
      this.recentsEl);
    this.refresh(); this.partition(); this.renderDets();
  }
  /** counts: {id: {items, px}} for the classes present in the frame (null = nothing known yet). */
  setCounts(counts) { this.counts = counts; this.partition(); }
  partition() {
    const counts = this.counts || {};
    const det = [...this.items.keys()].filter((id) => counts[id]);
    det.sort((a, b) => counts[b].items - counts[a].items || this.items.get(a)._c.name.localeCompare(this.items.get(b)._c.name));
    for (const it of this.items.values()) {
      const n = counts[it.dataset.id];
      it._cnt.hidden = !n;
      if (n) { it._cnt.textContent = String(n.items); it.title = `${it._c.raw_name} (${it._c.id}) - ${n.items} detections, ${n.px} px\n${HELP}`; }
      else it.title = `${it._c.raw_name} (${it._c.id})\n${HELP}`;
    }
    this.det.body.replaceChildren(...det.map((id) => this.items.get(id)));
    for (const g of this.groups.values()) {
      const rest = g._ids.filter((id) => !counts[id]);
      g._body.replaceChildren(...rest.map((id) => this.items.get(id)));
      g._cnt.textContent = String(rest.length);
    }
    this.nDet = det.length;
    this.setCount(this.det, det.length); this.setCount(this.undet, this.items.size - det.length);
    this.filter();
  }
  setCount(g, n) { const s = g.head.querySelector("small"); if (s) s.textContent = String(n); }
  filter() {
    const q = this.search.value.trim().toLowerCase();
    for (const it of this.items.values()) it.hidden = !!q && !it.dataset.q.includes(q);
    const vis = (ids) => ids.some((id) => !this.items.get(id).hidden);
    const counts = this.counts || {};
    for (const g of this.groups.values()) {
      const rest = g._ids.filter((id) => !counts[id]), any = vis(rest);
      g.hidden = !rest.length || (!!q && !any);
      if (q && any) setCollapsed(g, false);
      else if (!q) { const s = stored(g.dataset.gid); setCollapsed(g, s !== null ? s : true); }
    }
    const dIds = [...this.items.keys()].filter((id) => counts[id]);
    this.det.root.hidden = !!q && !vis(dIds);
    this.undet.root.hidden = !!q && ![...this.groups.values()].some((g) => !g.hidden);
  }
  /** list: [{k, px, bbox, seed}] of the selected class (null = no class selected); selK = k of the Inspect selection or null. */
  setDetections(list, selK) {
    if (list !== this.dets) this.detAll = false;
    this.dets = list; this.detSel = selK; this.renderDets();
  }
  renderDets() {
    const box = this.detEl, list = this.dets;
    if (!box) return;
    box.hidden = !list;
    if (!list) { box.replaceChildren(); return; }
    const LIM = 60, shown = this.detAll ? list : list.slice(0, LIM);
    const kids = [el("b", { class: "muted" }, `Detections (${list.length})`)];
    if (!list.length) kids.push(el("span", { class: "muted", "data-testid": "detections-empty" }, "0 detections in this frame"));
    for (const d of shown) {
      kids.push(el("button", { class: "pk-det" + (d.k === this.detSel ? " act" : ""), type: "button", "data-testid": "detection-item", dataset: { k: d.k },
        title: `Select detection #${d.k} and centre on it`, onclick: () => this.onDetection?.(d) }, `#${d.k} ${d.px} px`));
    }
    if (list.length > LIM) kids.push(el("button", { class: "pk-det more", type: "button", "data-testid": "detections-more", onclick: () => { this.detAll = !this.detAll; this.renderDets(); } },
      this.detAll ? "less" : `+${list.length - LIM} more`));
    box.replaceChildren(...kids);
  }
  focusSearch() { this.search?.focus(); this.search?.select(); }
  refresh() {
    const a = this.active == null ? null : S.byId.get(this.active), s = S.byId.get(this.secondary);
    this.curBox.hidden = !a;
    if (a) { this.curSw.style.background = rgbCss(a.color); this.curName.textContent = `${a.name} (${a.id})`; this.curBox.dataset.id = this.active; }
    else delete this.curBox.dataset.id;
    if (s) { this.secSw.style.background = rgbCss(s.color); this.secName.textContent = `${s.name} (${s.id})`; }
    for (const [id, it] of this.items) { it.classList.toggle("act", id === this.active); it.classList.toggle("sec", id === this.secondary); }
    this.protSel.value = this.protect >= 0 ? this.protect : "";
    this.recentsEl.replaceChildren(...this.recents.map((id, i) => {
      const c = S.byId.get(id);
      return el("button", { title: `${i + 1}: ${c.name}`, style: { background: rgbCss(c.color) }, onclick: () => this.setActive(id) }, el("i", {}, String(i + 1)));
    }));
  }
  remember(id) {
    this.recents = [id, ...this.recents.filter((x) => x !== id)].slice(0, 9);
    try { localStorage.setItem("lab.recents", JSON.stringify(this.recents)); } catch { /* ignore */ }
  }
  pick(id) { this.remember(id); this.setActive(id); }
  setActive(id) { if (id !== null && !S.byId.has(id)) return; this.active = id; this.refresh(); this.onChange?.(); }
  setSecondary(id) { if (!S.byId.has(id)) return; this.secondary = id; this.refresh(); this.onChange?.(); }
  setProtect(id) { this.protect = id; this.refresh(); this.onChange?.(); }
  byDigit(n) { const id = this.recents[n - 1]; if (id !== undefined) this.setActive(id); }
  swap() {
    if (this.active == null) this.active = this.secondary; else [this.active, this.secondary] = [this.secondary, this.active];
    this.refresh(); this.onChange?.();
  }
}
const HELP = "click: paint class (again: clear) - right-click: secondary (Erase) - double-click: protect";
