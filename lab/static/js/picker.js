// Class picker: grouped by hier, search, recents (digits 1-9), secondary class, protect dropdown.
import { el, rgbCss, debounce } from "./util.js";
import { S, on } from "./store.js";

export class Picker {
  constructor(host, onChange) {
    this.host = host; this.onChange = onChange;
    this.active = 0; this.secondary = 0; this.protect = -1;
    this.recents = [];
    try { this.recents = JSON.parse(localStorage.getItem("lab.recents") || "[]"); } catch { /* ignore */ }
    this.items = new Map();
    this.build();
    on("taxonomy", () => this.build());
  }
  build() {
    const tax = S.tax;
    if (!tax.length) return;
    if (!S.byId.has(this.active)) this.active = tax[0].id;
    if (!S.byId.has(this.secondary)) this.secondary = tax[0].id;
    this.recents = this.recents.filter((id) => S.byId.has(id)).slice(0, 9);

    this.curBox = el("div", { class: "pk-cur", "data-testid": "active-class" }); this.curSw = el("span", { class: "sw big" }); this.curBox.append(this.curSw, this.curName = el("b"));     this.secSw = el("span", { class: "sw" }); this.secName = el("span", { class: "muted" });
    this.search = el("input", { "data-testid": "picker-search", type: "search", placeholder: "Search classes (/)", autocomplete: "off" });
    this.search.addEventListener("input", debounce(() => this.filter(), 60));
    this.search.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { const first = [...this.items.values()].find((n) => !n.hidden); if (first) { this.pick(Number(first.dataset.id), e); this.search.blur(); } }
      else if (e.key === "Escape") { this.search.value = ""; this.filter(); this.search.blur(); }
    });
    this.protSel = el("select", { title: "Protect source class: tools only change pixels of this class (Shift+Alt+click on canvas)" },
      el("option", { value: "" }, "none"),
      ...[...tax].sort((a, b) => a.name.localeCompare(b.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`)));
    this.protSel.value = this.protect >= 0 ? this.protect : "";
    this.protSel.setAttribute("data-testid", "protect-select");
    this.protSel.addEventListener("change", () => { this.protect = this.protSel.value === "" ? -1 : Number(this.protSel.value); this.protSel.blur(); this.onChange?.(); });
    this.recentsEl = el("div", { class: "pk-recents" });
    this.list = el("div", { class: "pk-list" });

    const groups = new Map();
    for (const c of tax) { if (!groups.has(c.hier)) groups.set(c.hier, []); groups.get(c.hier).push(c); }
    this.items.clear();
    for (const [hier, cs] of groups) {
      const d = el("details", { class: "pk-group", open: true }, el("summary", {}, `${hier || "(none)"} (${cs.length})`));
      for (const c of cs) {
        const it = el("div", { class: "pk-item", "data-testid": "picker-item", dataset: { id: c.id }, title: `${c.raw_name}\nclick: paint class - right-click: secondary (Erase) - double-click: protect` },
          el("span", { class: "sw", style: { background: rgbCss(c.color) } }), el("span", {}, c.name), el("span", { class: "id mono" }, String(c.id)));
        it.dataset.q = `${c.name} ${c.raw_name} ${c.id}`.toLowerCase();
        it.addEventListener("click", (e) => this.pick(c.id, e));
        it.addEventListener("contextmenu", (e) => { e.preventDefault(); this.setSecondary(c.id); });
        it.addEventListener("dblclick", () => this.setProtect(c.id));
        d.append(it); this.items.set(c.id, it);
      }
      this.list.append(d);
    }
    this.host.replaceChildren(
      el("div", { class: "pk-head" },
        this.curBox,
        el("div", { class: "pk-cur", title: "Secondary class for Erase tool (right-click a class, X swaps)" }, el("span", { class: "muted" }, "Erase to"), this.secSw, this.secName),
        el("label", { class: "muted" }, "Protect source ", this.protSel),
        this.recentsEl, this.search),
      this.list);
    this.refresh();
  }
  filter() {
    const q = this.search.value.trim().toLowerCase();
    for (const it of this.items.values()) it.hidden = !!q && !it.dataset.q.includes(q);
    for (const d of this.list.children) { d.hidden = ![...d.querySelectorAll(".pk-item")].some((n) => !n.hidden); if (q) d.open = true; }
  }
  focusSearch() { this.search?.focus(); this.search?.select(); }
  refresh() {
    const a = S.byId.get(this.active), s = S.byId.get(this.secondary);
    if (!a) return;
    this.curSw.style.background = rgbCss(a.color); this.curName.textContent = `${a.name} (${a.id})`;
    this.secSw.style.background = rgbCss(s.color); this.secName.textContent = `${s.name} (${s.id})`;
    for (const [id, it] of this.items) { it.classList.toggle("act", id === this.active); it.classList.toggle("sec", id === this.secondary); }
    this.protSel.value = this.protect >= 0 ? this.protect : "";
    this.curBox.dataset.id = this.active;
    this.recentsEl.replaceChildren(...this.recents.map((id, i) => {
      const c = S.byId.get(id);
      return el("button", { title: `${i + 1}: ${c.name}`, style: { background: rgbCss(c.color) }, onclick: () => this.setActive(id) }, el("i", {}, String(i + 1)));
    }));
  }
  remember(id) {
    this.recents = [id, ...this.recents.filter((x) => x !== id)].slice(0, 9);
    try { localStorage.setItem("lab.recents", JSON.stringify(this.recents)); } catch { /* ignore */ }
  }
  /** List/eyedropper pick: sets active and pushes to recents. */
  pick(id) { this.remember(id); this.setActive(id); }
  setActive(id) { if (!S.byId.has(id)) return; this.active = id; this.refresh(); this.onChange?.(); }
  setSecondary(id) { if (!S.byId.has(id)) return; this.secondary = id; this.refresh(); this.onChange?.(); }
  setProtect(id) { this.protect = id; this.refresh(); this.onChange?.(); }
  /** Digit key: choose recents[n-1] without reordering. */
  byDigit(n) { const id = this.recents[n - 1]; if (id !== undefined) this.setActive(id); }
  swap() { [this.active, this.secondary] = [this.secondary, this.active]; this.refresh(); this.onChange?.(); }
}
