// Router (tabs + URL hash), status bar, help overlay, global keyboard dispatch.
import { $, $$, el, toast, typing, status } from "./js/util.js";
import { S, loadTaxonomy, loadMeta, on } from "./js/store.js";
import * as browser from "./js/browser.js";
import * as editor from "./js/editor.js";
import * as taxonomy from "./js/taxonomy.js";
import * as rules from "./js/rules.js";

const TABS = ["browse", "edit", "taxonomy", "rules"];
let curTab = null, curHash = "", lastEdit = "";

const HELP = [
  ["Global", [["?", "this help"], ["Esc", "close dialogs / cancel"]]],
  ["Browse", [["Arrows", "move selection"], ["Shift/Cmd-click", "extend / toggle selection"], ["1 2 3 4", "tag ok / fix / drop / todo"], ["Enter or dbl-click", "open in editor"]]],
  ["Editor: tools", [["B", "brush (size 1-64)"], ["E", "erase to secondary class"], ["R", "region relabel (Shift = whole class)"], ["L", "polygon / lasso (Enter or dbl-click closes, Esc cancels)"],
    ["S", "SAM: left/right click = +/- point, drag = box, Tab cycles, Enter applies, Esc clears"], ["[  ]", "brush size"], ["Q", "round / square"], ["X", "swap active and secondary class"]]],
  ["Editor: classes", [["1-9", "pick recent class"], ["/", "search classes"], ["Alt+click", "eyedropper (pick class under cursor)"], ["Shift+Alt+click", "protect class under cursor"],
    ["Right-click list item", "set secondary (Erase) class"], ["Double-click list item", "set protect class"]]],
  ["Editor: view", [["Wheel", "zoom to cursor (fit - 32x)"], ["Space-drag / middle-drag", "pan"], ["F", "fit"], ["0", "actual pixels (1:1)"], ["+ / -", "zoom"], ["M", "toggle mask"], ["O", "outline mode"], ["G", "pixel grid (shown from 8x)"]]],
  ["Editor: file", [["Ctrl/Cmd+Z", "undo"], ["Ctrl/Cmd+Shift+Z or Y", "redo"], ["Ctrl/Cmd+S", "save"], ["N / P", "next / previous in the Browse filter (autosaves)"], ["Ctrl/Cmd+1..4", "status ok / fix / drop / todo"]]],
];
function buildHelp() {
  $("#help .help-card").replaceChildren(el("h2", {}, "Keyboard shortcuts"),
    el("div", { class: "cols" }, HELP.map(([h, items]) => el("section", {}, el("h3", {}, h), el("dl", {}, items.map(([k, d]) => [el("dt", {}, k), el("dd", {}, d)]).flat())))),
    el("p", { class: "hint" }, "Click anywhere or press Esc to close."));
}
const toggleHelp = (v) => { $("#help").hidden = v === undefined ? !$("#help").hidden : !v; };

function parseHash() {
  const raw = location.hash.slice(1) || "browse", q = raw.indexOf("?");
  const path = q < 0 ? raw : raw.slice(0, q), qs = q < 0 ? "" : raw.slice(q + 1);
  const [tab, ...rest] = path.split("/");
  return { tab: TABS.includes(tab) ? tab : "browse", stem: decodeURIComponent(rest.join("/")), qs };
}

async function route() {
  const r = parseHash(), hash = location.hash;
  if (curTab === "edit" && !(r.tab === "edit" && r.stem)) {
    if (!(await editor.leave())) { history.replaceState(null, "", curHash); return; }
  }
  if (r.tab === "edit" && !r.stem) {
    if (lastEdit) { location.hash = lastEdit; return; }
    toast("Open a frame from Browse first", "error", 2500); location.hash = "#browse"; return;
  }
  curTab = r.tab; curHash = hash || "#browse";
  $$(".page").forEach((p) => (p.hidden = p.id !== `page-${r.tab}`));
  $$("#tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === r.tab));
  if (r.tab === "browse") browser.show(r.qs);
  else if (r.tab === "edit") { lastEdit = curHash; editor.onShow(); editor.show(r.stem, r.qs); }
  else if (r.tab === "taxonomy") taxonomy.show();
  else if (r.tab === "rules") rules.show();
  status.set("tool", r.tab === "edit" ? "" : ""); status.set("zoom", ""); status.set("cursor", "");
}

function renderMeta() {
  const m = S.meta;
  if (!m) { status.set("sam", "backend unreachable"); return; }
  status.set("version", `v${m.version}`); status.set("dataset", m.dataset);
  const sam = $("#sb-sam"), ok = !!m.sam?.available;
  sam.textContent = ok ? `SAM ready${m.sam.loaded ? " (loaded)" : ""}` : "SAM unavailable";
  sam.className = ok ? "ok" : "off"; sam.dataset.available = String(ok);
}

function onKeyDown(e) {
  if (!$("#modal").hidden) return;
  if (!$("#help").hidden) { if (e.key === "Escape" || e.key === "?") { toggleHelp(false); e.preventDefault(); } return; }
  const mod = e.ctrlKey || e.metaKey;
  if (typing(e.target)) { if (e.key === "Escape") e.target.blur(); else if (!(mod && curTab === "edit" && e.key.toLowerCase() === "s")) return; }
  if (e.key === "?" && !mod) { toggleHelp(true); e.preventDefault(); return; }
  let handled = false;
  if (curTab === "browse") handled = browser.onKeyGlobal(e);
  else if (curTab === "edit") handled = editor.onKey(e);
  if (handled) e.preventDefault();
}

async function main() {
  buildHelp();
  $("#helpBtn").addEventListener("click", () => toggleHelp(true));
  $("#help").addEventListener("mousedown", () => toggleHelp(false));
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", (e) => { if (curTab === "edit" && editor.onKeyUp(e)) e.preventDefault(); });
  window.addEventListener("hashchange", route);
  on("meta", renderMeta);
  await Promise.all([loadMeta(), loadTaxonomy().catch(() => toast("Could not load the taxonomy; is the backend up?"))]);
  renderMeta();
  setInterval(loadMeta, 30000);
  browser.init(); editor.init(); taxonomy.init(); rules.init();
  await route();
}
main();
