// Class picker: compact chips grouped by hier (each hier collapsible), search, recents (digits 1-9), secondary class, protect.
import { el, rgbCss, debounce } from "./util.js";
import { S, on } from "./store.js";
import { makeGroup, applyDefault, setCollapsed, stored } from "./groups.js";

export class Picker {
  constructor(host, protHost, onChange) {
    this.host = host; this.protHost = protHost; this.onChange = onChange;
    this.active = 0; this.secondary = 0; this.protect = -1;
    this.recents = [];
    try { this.recents = JSON.parse(localStorage.getItem("lab.recents") || "[]"); } catch { /* ignore */ }
    this.items = new Map(); this.groups = new Map(); this.present = null;
    this.build();
    on("taxonomy", () => this.build());
  }
  build() {
    const tax = S.tax;
    if (!tax.length) return;
    if (!S.byId.has(this.active)) this.active = tax[0].id;
    if (!S.byId.has(this.secondary)) this.secondary = tax[0].id;
    this.recents = this.recents.filter((id) => S.byId.has(id)).slice(0, 9);

    this.curBox = el("div", { class: "pk-cur", "data-testid": "active-class" }, this.curSw = el("span", { class: "sw big" }), this.curName = el("b"));
    this.secSw = el("span", { class: "sw" }); this.secName = el("span", { class: "muted" });
    this.search = el("input", { "data-testid": "picker-search", type: "search", placeholder: "Search classes (/)", autocomplete: "off" });
    this.search.addEventListener("input", debounce(() => this.filter(), 60));
    this.search.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { const first = [...this.items.values()].find((n) => !n.hidden); if (first) { this.pick(Number(first.dataset.id)); this.search.blur(); } }
      else if (e.key === "Escape") { this.search.value = ""; this.filter(); this.search.blur(); }
    });
    this.protSel = el("select", { "data-testid": "protect-select", title: "Protect source class: tools only change pixels of this class (Shift+Alt+click on the image)" },
      el("option", { value: "" }, "none"),
      ...[...tax].sort((a, b) => a.name.localeCompare(b.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`)));
    this.protSel.addEventListener("change", () => { this.protect = this.protSel.value === "" ? -1 : Number(this.protSel.value); this.protSel.blur(); this.onChange?.(); });
    this.recentsEl = el("div", { class: "pk-recents" });
    this.list = el("div", { class: "pk-list" });

    const hiers = new Map();
    for (const c of tax) { if (!hiers.has(c.hier)) hiers.set(c.hier, []); hiers.get(c.hier).push(c); }
    this.items.clear(); this.groups.clear();
    for (const [hier, cs] of hiers) {
      const g = makeGroup(`hier-${hier || "none"}`, hier || "(none)", { count: `${cs.length}` });
      for (const c of cs) {
        const it = el("span", { class: "pk-item", "data-testid": "picker-item", dataset: { id: c.id }, title: `${c.raw_name} (${c.id})\nclick: paint class - right-click: secondary (Erase) - double-click: protect` },
          el("span", { class: "sw", style: { background: rgbCss(c.color) } }), el("span", {}, c.name));
        it.dataset.q = `${c.name} ${c.raw_name} ${c.id}`.toLowerCase();
        it.addEventListener("click", () => this.pick(c.id));
        it.addEventListener("contextmenu", (e) => { e.preventDefault(); this.setSecondary(c.id); });
        it.addEventListener("dblclick", () => this.setProtect(c.id));
        g.body.append(it); this.items.set(c.id, it);
      }
      g.root.dataset.hier = hier; g.root._ids = cs.map((c) => c.id);
      this.groups.set(hier, g.root); this.list.append(g.root);
    }
    this.host.replaceChildren(
      el("div", { class: "pk-head" }, this.curBox,
        el("div", { class: "pk-cur", title: "Secondary class for the Erase tool (right-click a chip, X swaps)" }, el("span", { class: "muted" }, "Erase to"), this.secSw, this.secName),
        this.recentsEl, this.search),
      this.list);
    this.protHost.replaceChildren(el("label", { class: "muted" }, "Only change pixels of ", this.protSel));
    this.refresh(); this.applyPresence();
  }
  /** Hier groups without pixels in the current frame start collapsed (unless the user chose a state). */
  setPresent(ids) { this.present = ids; this.applyPresence(); }
  applyPresence() {
    if (!this.present) return;
    for (const [id, it] of this.items) it.classList.toggle("absent", !this.present.has(id));
    for (const g of this.groups.values()) applyDefault(g, !g._ids.some((id) => this.present.has(id)));
  }
  filter() {
    const q = this.search.value.trim().toLowerCase();
    for (const it of this.items.values()) it.hidden = !!q && !it.dataset.q.includes(q);
    for (const g of this.groups.values()) {
      const any = g._ids.some((id) => !this.items.get(id).hidden);
      g.hidden = !any;
      if (q && any) setCollapsed(g, false);
      else if (!q) { const s = stored(g.dataset.gid); setCollapsed(g, s !== null ? s : !!this.present && !g._ids.some((id) => this.present.has(id))); }
    }
  }
  focusSearch() { this.search?.focus(); this.search?.select(); }
  refresh() {
    const a = S.byId.get(this.active), s = S.byId.get(this.secondary);
    if (!a) return;
    this.curSw.style.background = rgbCss(a.color); this.curName.textContent = `${a.name} (${a.id})`;
    this.secSw.style.background = rgbCss(s.color); this.secName.textContent = `${s.name} (${s.id})`;
    this.curBox.dataset.id = this.active;
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
  setActive(id) { if (!S.byId.has(id)) return; this.active = id; this.refresh(); this.onChange?.(); }
  setSecondary(id) { if (!S.byId.has(id)) return; this.secondary = id; this.refresh(); this.onChange?.(); }
  setProtect(id) { this.protect = id; this.refresh(); this.onChange?.(); }
  byDigit(n) { const id = this.recents[n - 1]; if (id !== undefined) this.setActive(id); }
  swap() { [this.active, this.secondary] = [this.secondary, this.active]; this.refresh(); this.onChange?.(); }
}
