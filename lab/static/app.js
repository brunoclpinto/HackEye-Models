// Router (tabs + URL hash), single-page groups, status bar, help overlay, global keyboard dispatch.
// Hash: #<tab>[/<stem>][?<browse filter>], tab = edit | taxonomy | rules (default edit).
import { $, $$, el, toast, typing, status } from "./js/util.js";
import { S, loadTaxonomy, loadMeta, on } from "./js/store.js";
import { nav, hashFor, replaceHash } from "./js/nav.js";
import { bindAll } from "./js/groups.js";
import * as browser from "./js/browser.js";
import * as editor from "./js/editor.js";
import * as taxonomy from "./js/taxonomy.js";
import * as rules from "./js/rules.js";

const TABS = ["edit", "taxonomy", "rules"], TITLES = { edit: "Work: Edit", taxonomy: "Work: Taxonomy", rules: "Work: Rules" };
const TAGS = { Digit1: "ok", Digit2: "fix", Digit3: "drop", Digit4: "todo" };

const HELP = [
  ["Frames", [["Left / Right, P / N", "previous / next frame in the Browse filter (autosaves)"], ["Home / End", "first / last frame"], ["Shift+1 2 3 4", "tag ok / fix / drop / todo"],
    ["Click / drag / wheel on the carousel", "open frame / scroll the strip"]]],
  ["Tools", [["B", "brush (size 1-64)"], ["E", "erase to secondary class"], ["R", "region relabel (Shift = whole class)"], ["L", "polygon / lasso (Enter or dbl-click closes, Esc cancels)"],
    ["S", "SAM: left/right click = +/- point, drag = box, Tab cycles, Enter applies, Esc clears"], ["[  ]", "brush size"], ["Q", "round / square"], ["X", "swap active and secondary class"]]],
  ["Classes", [["1-9", "pick recent class"], ["/", "search classes"], ["Alt+click", "eyedropper (pick class under cursor)"], ["Shift+Alt+click", "protect class under cursor"],
    ["Right-click a chip", "set secondary (Erase) class"], ["Double-click a chip", "set protect class"]]],
  ["View", [["Wheel", "zoom to cursor (fit - 32x)"], ["Space-drag / middle-drag", "pan"], ["F", "fit"], ["0", "actual pixels (1:1)"], ["+ / -", "zoom"], ["M", "toggle mask"], ["O", "outline mode"], ["G", "pixel grid (shown from 8x)"],
    ["Drag the bar under the image", "resize the image (double-click resets)"]]],
  ["File", [["Ctrl/Cmd+Z", "undo"], ["Ctrl/Cmd+Shift+Z or Y", "redo"], ["Ctrl/Cmd+S", "save"], ["?", "this help"], ["Esc", "close dialogs / cancel"]]],
];
function buildHelp() {
  $("#help .help-card").replaceChildren(el("h2", {}, "Keyboard shortcuts"),
    el("div", { class: "cols" }, HELP.map(([h, items]) => el("section", {}, el("h3", {}, h), el("dl", {}, items.map(([k, d]) => [el("dt", {}, k), el("dd", {}, d)]).flat())))),
    el("p", { class: "hint" }, "Every group has a chevron header: click it to collapse or expand (remembered). Click anywhere or press Esc to close."));
}
const toggleHelp = (v) => { $("#help").hidden = v === undefined ? !$("#help").hidden : !v; };

function parseHash() {
  const raw = location.hash.slice(1), q = raw.indexOf("?");
  const path = q < 0 ? raw : raw.slice(0, q), qs = q < 0 ? "" : raw.slice(q + 1);
  const [tab, ...rest] = path.split("/");
  return { tab: TABS.includes(tab) ? tab : "edit", stem: decodeURIComponent(rest.join("/")), qs, bare: !TABS.includes(tab) };
}

let prevTab = null, routing = Promise.resolve();
function route() { routing = routing.then(doRoute).catch((e) => console.error(e)); return routing; }
async function doRoute() {
  const r = parseHash();
  const tabChanged = r.tab !== prevTab;
  nav.tab = r.tab; nav.qs = r.qs;
  await browser.setFilter(r.qs);
  let stem = r.stem;
  if (!stem) stem = nav.stem || browser.firstStem();
  nav.stem = stem;
  replaceHash(hashFor(stem, r.tab, r.qs));          // keep the URL complete: tab + frame + filter
  $$("#tabs a").forEach((a) => { a.classList.toggle("on", a.dataset.tab === r.tab); a.href = hashFor(nav.stem, a.dataset.tab, nav.qs); });
  if (tabChanged) {
    $$(".pane").forEach((p) => (p.hidden = p.id !== `pane-${r.tab}`));
    $("#workTitle").textContent = TITLES[r.tab];
    if (prevTab === "taxonomy") taxonomy.hide();
    if (r.tab === "taxonomy") taxonomy.show(); else if (r.tab === "rules") rules.show();
    prevTab = r.tab;
  }
  if (stem && stem !== editor.currentStem()) await editor.show(stem, r.qs);
}

function renderMeta() {
  const m = S.meta;
  if (!m) { status.set("sam", "backend unreachable"); return; }
  status.set("version", /^v/i.test(m.version) ? m.version : `v${m.version}`); status.set("dataset", m.dataset);
  const sam = $("#sb-sam"), ok = !!m.sam?.available;
  sam.textContent = ok ? `SAM ready${m.sam.loaded ? " (loaded)" : ""}` : "SAM unavailable";
  sam.className = ok ? "ok" : "off"; sam.dataset.available = String(ok);
}

function onKeyDown(e) {
  if (!$("#modal").hidden) return;
  if (!$("#help").hidden) { if (e.key === "Escape" || e.key === "?") { toggleHelp(false); e.preventDefault(); } return; }
  const mod = e.ctrlKey || e.metaKey;
  if (typing(e.target)) { if (e.key === "Escape") e.target.blur(); else if (!(mod && e.key.toLowerCase() === "s")) return; }
  if (e.key === "?" && !mod) { toggleHelp(true); e.preventDefault(); return; }
  if (!mod && !e.altKey && e.shiftKey && TAGS[e.code]) { editor.setStatus(TAGS[e.code]); e.preventDefault(); return; }
  if (!mod && !e.altKey && !e.shiftKey && (e.key === "Home" || e.key === "End")) { browser.jump(e.key === "Home" ? "first" : "last"); e.preventDefault(); return; }
  if (editor.onKey(e)) e.preventDefault();
}

async function main() {
  buildHelp();
  bindAll();
  $("#helpBtn").addEventListener("click", () => toggleHelp(true));
  $("#help").addEventListener("mousedown", () => toggleHelp(false));
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", (e) => { if (editor.onKeyUp(e)) e.preventDefault(); });
  window.addEventListener("hashchange", route);
  on("meta", renderMeta);
  await Promise.all([loadMeta(), loadTaxonomy().catch(() => toast("Could not load the taxonomy; is the backend up?"))]);
  renderMeta();
  setInterval(loadMeta, 30000);
  browser.init(); editor.init(); taxonomy.init(); rules.init();
  await route();
}
main();
