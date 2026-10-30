// Taxonomy tab: class table (create/split/edit/delete custom classes) + merge-map builder with live preview.
import { api } from "./api.js";
import { $, el, toast, debounce, dialog, confirmDialog, rgbCss, rgbHex, hexRgb, pct, fmtPx, pollJob, renderJob } from "./util.js";
import { S, loadTaxonomy, on } from "./store.js";
import * as editor from "./editor.js";

const IGN = "__ignore__", NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const T = { selGroups: new Set(), savedSig: "", checked: new Set(), groups: {}, colors: {}, preview: null, filter: "", mapName: "" };

export function init() {
  $("#tFilter").addEventListener("input", debounce((e) => { T.filter = e.target.value.trim().toLowerCase(); renderTable(); }, 80));
  $("#tNew").addEventListener("click", newClass);
  $("#tSplit").addEventListener("click", splitClass);
  $("#tEdit").addEventListener("click", editClass);
  $("#tDelete").addEventListener("click", deleteClass);
  $("#tMerge").addEventListener("click", mergeChecked);
  $("#tVoid").addEventListener("click", voidChecked);
  $("#tLoad").addEventListener("click", loadMap);
  $("#tMapSel").addEventListener("change", loadMap);
  $("#tIdentity").addEventListener("click", () => { T.groups = {}; T.colors = {}; T.mapName = ""; changed(); });
  $("#tSaveMap").addEventListener("click", () => saveMap());
  $("#tExport").addEventListener("click", exportDataset);
  on("taxonomy", () => { renderTable(); schedulePreview(); });
  $("#tName").placeholder = "HackEye<yy.major.minor>-<N>";
}

let active = false;
export async function show() {
  active = true; if (T.preview) editor.setViewPreview(T.preview);
  await refreshMapList();
  renderTable(); renderGroups();
  if (!T.preview) schedulePreview();
}

export function hide() { if (active) { active = false; editor.setViewPreview(null); } }

async function refreshMapList() {
  try {
    const maps = await api("/api/maps", { quiet: true }), sel = $("#tMapSel"), cur = sel.value;
    sel.replaceChildren(el("option", { value: "" }, "(choose)"), ...maps.map((m) => el("option", { value: m.name }, `${m.name} (${m.n_classes})`)));
    sel.value = cur;
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------- class table
const rawOf = (id) => S.byId.get(id)?.raw_name;
function groupOfId(id) {
  const c = T.preview?.classes.find((k) => k.members.includes(id));
  return c ? (c.void ? "void" : c.members.length > 1 || c.name !== S.byId.get(id)?.name ? c.name : "") : "";
}
function renderTable() {
  const total = S.tax.reduce((a, c) => a + (c.pixels || 0), 0) || 1;
  const rows = S.tax.filter((c) => !T.filter || `${c.name} ${c.raw_name} ${c.hier} ${c.id}`.toLowerCase().includes(T.filter));
  const tbl = el("table", { class: "tbl" },
    el("thead", {}, el("tr", {}, ...["", "", "id", "name", "hier", "frames", "pixel share", "map group", ""].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.map((c) => {
      const tr = el("tr", { class: T.checked.has(c.id) ? "chk" : "", title: c.raw_name, "data-testid": "class-row", dataset: { id: c.id, custom: String(!!c.custom) } },
        el("td", {}, el("input", { type: "checkbox", title: "Check this class", checked: T.checked.has(c.id), style: { pointerEvents: "none" } })),
        el("td", {}, el("span", { class: "sw", style: { background: rgbCss(c.color) } })),
        el("td", { class: "mono" }, String(c.id)), el("td", {}, c.name), el("td", { class: "muted" }, c.hier),
        el("td", { class: "num mono" }, String(c.frames ?? 0)), el("td", { class: "num mono" }, pct((c.pixels || 0) / total)),
        el("td", { class: "muted" }, groupOfId(c.id)),
        el("td", {}, c.custom ? el("span", { class: "badge custom" }, c.parent != null ? `custom < ${S.byId.get(c.parent)?.name ?? c.parent}` : "custom") : ""));
      tr.addEventListener("click", () => {
        if (T.checked.has(c.id)) T.checked.delete(c.id); else T.checked.add(c.id);
        tr.classList.toggle("chk", T.checked.has(c.id)); tr.querySelector("input").checked = T.checked.has(c.id); syncButtons();
      });
      return tr;
    })));
  const host = $("#tTable"), pg = $("#page"), st = pg.scrollTop;
  host.replaceChildren(tbl); pg.scrollTop = st;
  syncButtons();
}
function syncButtons() {
  const one = T.checked.size === 1 ? S.byId.get([...T.checked][0]) : null;
  $("#tSplit").disabled = !one; $("#tEdit").disabled = !(one && one.custom); $("#tDelete").disabled = !(one && one.custom);
  $("#tMerge").disabled = T.checked.size < 1 && !T.selGroups.size; $("#tVoid").disabled = T.checked.size < 1;
  $("#tMerge").textContent = `Merge selected (${T.checked.size + T.selGroups.size})`;
}

// ---------------------------------------------------------------- class dialogs
const hiers = () => [...new Set(S.tax.map((c) => c.hier))].sort();
async function newClass() {
  const v = await dialog("New class", (b) => {
    const name = el("input", { type: "text", "data-testid": "dlg-name", placeholder: "e.g. Speed bump", title: "Name of the new class" }), hier = el("input", { type: "text", "data-testid": "dlg-hier", list: "hierList", value: "object", title: "Hierarchy group the class is listed under" });
    const dl = el("datalist", { id: "hierList" }, hiers().map((h) => el("option", { value: h })));
    const color = el("input", { type: "color", "data-testid": "dlg-color", value: "#ff8800", title: "Class colour" });
    const parent = el("select", { title: "Parent class (the new class is carved out of it)" }, el("option", { value: "" }, "(none)"), [...S.tax].sort((a, c) => a.name.localeCompare(c.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`)));
    b.append(el("label", {}, "Name", name), el("label", {}, "Hier group", hier, dl), el("label", {}, "Colour", color), el("label", {}, "Parent (optional)", parent));
    return () => { if (!name.value.trim()) { name.focus(); return undefined; } return { name: name.value.trim(), hier: hier.value.trim() || "object", color: hexRgb(color.value), parent: parent.value === "" ? undefined : Number(parent.value) }; };
  }, "Create");
  if (!v) return;
  try { await api("/api/taxonomy/classes", { method: "POST", body: v }); await loadTaxonomy(); toast(`Created ${v.name}`, "ok", 2000); } catch { /* toasted */ }
}
async function splitClass() {
  const p = S.byId.get([...T.checked][0]);
  if (!p) return;
  const v = await dialog(`Split "${p.name}" into children`, (b) => {
    const rows = el("div", { class: "dlg-body" });
    const add = (n = "", c = p.color) => {
      const row = el("div", { class: "crow" }, el("input", { type: "text", "data-testid": "dlg-child-name", placeholder: "child name", value: n, title: "Name of the child class" }), el("input", { type: "color", value: rgbHex(c), title: "Child class colour" }),
        el("button", { type: "button", title: "Remove this child", onclick: () => row.remove() }, "x"));
      rows.append(row);
    };
    const jitter = (i) => p.color.map((x, k) => Math.max(0, Math.min(255, x + ((i + k) % 3 - 1) * 40)));
    add("", jitter(0)); add("", jitter(1));
    b.append(el("p", { class: "hint" }, "Creates custom classes under the parent; no pixels move. Paint them in the editor."), rows, el("button", { type: "button", title: "Add another child class", onclick: () => add("", jitter(rows.children.length)) }, "+ child"));
    return () => {
      const ch = [...rows.children].map((r) => ({ name: r.children[0].value.trim(), color: hexRgb(r.children[1].value) })).filter((c) => c.name);
      return ch.length ? ch : undefined;
    };
  }, "Split");
  if (!v) return;
  try { await api(`/api/taxonomy/classes/${p.id}/split`, { method: "POST", body: { children: v } }); await loadTaxonomy(); toast(`Split ${p.name}`, "ok", 2000); } catch { /* toasted */ }
}
async function editClass() {
  const c = S.byId.get([...T.checked][0]);
  if (!c?.custom) return;
  const v = await dialog(`Edit ${c.name}`, (b) => {
    const name = el("input", { type: "text", "data-testid": "dlg-name", value: c.name, title: "Class name" }), hier = el("input", { type: "text", "data-testid": "dlg-hier", value: c.hier, list: "hierList", title: "Hierarchy group" });
    const dl = el("datalist", { id: "hierList" }, hiers().map((h) => el("option", { value: h })));
    const color = el("input", { type: "color", value: rgbHex(c.color), title: "Class colour" });
    b.append(el("label", {}, "Name", name), el("label", {}, "Hier group", hier, dl), el("label", {}, "Colour", color));
    return () => (name.value.trim() ? { name: name.value.trim(), hier: hier.value.trim(), color: hexRgb(color.value) } : undefined);
  }, "Save");
  if (!v) return;
  try { await api(`/api/taxonomy/classes/${c.id}`, { method: "PATCH", body: v }); await loadTaxonomy(); } catch { /* toasted */ }
}
async function deleteClass() {
  const c = S.byId.get([...T.checked][0]);
  if (!c?.custom) return;
  let url = `/api/taxonomy/classes/${c.id}`;
  if ((c.pixels || 0) > 0) {
    const to = await dialog(`Delete ${c.name}: remap ${fmtPx(c.pixels)} px to`, (b) => {
      const sel = el("select", { title: "Class that receives this class's pixels" }, [...S.tax].filter((k) => k.id !== c.id).sort((a, k) => a.name.localeCompare(k.name)).map((k) => el("option", { value: k.id }, `${k.name} (${k.id})`)));
      if (c.parent != null) sel.value = c.parent;
      b.append(el("p", { class: "hint" }, "Every pixel of this class is relabelled in all frames (a rule job), then the class is deleted."), el("label", {}, "Remap to", sel));
      return () => Number(sel.value);
    }, "Remap and delete");
    if (to === null) return;
    url += `?remap_to=${to}`;
  } else if (!(await confirmDialog(`Delete class "${c.name}"?`, "Delete"))) return;
  try {
    const r = await api(url, { method: "DELETE" });
    T.checked.delete(c.id);
    if (r.job) {
      const host = $("#tJob");
      const job = await pollJob(api, r.job, (j) => renderJob(host, j, "Delete class"));
      if (job.status !== "done") { toast("Delete job failed"); return; }
    }
    await loadTaxonomy(); toast(`Deleted ${c.name}`, "ok", 2000);
  } catch { /* toasted */ }
}

// ---------------------------------------------------------------- map builder
function removeEverywhere(raws) {
  const rs = new Set(raws);
  for (const k of Object.keys(T.groups)) { T.groups[k] = T.groups[k].filter((m) => !rs.has(m)); if (!T.groups[k].length) delete T.groups[k]; }
}
async function mergeChecked() {
  const gnames = [...T.selGroups].filter((g) => T.groups[g] && g !== IGN);
  const ids = [...T.checked];
  if (!ids.length && !gnames.length) return;
  const gRaws = gnames.flatMap((g) => T.groups[g]);
  const first = S.byId.get(ids[0]) ?? S.byRaw.get(gRaws[0]) ?? S.tax[0];
  const v = await dialog(`Merge ${ids.length + gnames.length} item${ids.length + gnames.length > 1 ? "s" : ""}`, (b) => {
    const name = el("input", { type: "text", "data-testid": "dlg-name", value: ids.length === 1 && !T.selGroups.size ? first.name : "", placeholder: "merged class name", title: "Name of the merged class" });
    const color = el("input", { type: "color", "data-testid": "dlg-color", value: rgbHex(first.color), title: "Merged class colour" });
    b.append(el("p", { class: "hint" }, [...gnames, ...ids.map((i) => S.byId.get(i).name)].join(", ")), el("label", {}, "Merged name (an existing group name adds to it)", name), el("label", {}, "Colour (display in this builder)", color));
    return () => { const n = name.value.trim(); if (!n || n === IGN) { name.focus(); return undefined; } return { name: n, color: hexRgb(color.value) }; };
  }, "Merge");
  if (!v) return;
  const raws = [...new Set([...gRaws, ...ids.map(rawOf).filter(Boolean)])];
  removeEverywhere(raws);
  T.selGroups.clear();
  T.groups[v.name] = [...(T.groups[v.name] || []), ...raws]; T.colors[v.name] = v.color;
  T.checked.clear(); changed();
}
function voidChecked() {
  const raws = [...T.checked].map(rawOf).filter(Boolean); if (!raws.length) return;
  removeEverywhere(raws); T.groups[IGN] = [...(T.groups[IGN] || []), ...raws]; T.checked.clear(); changed();
}
async function loadMap() {
  const name = $("#tMapSel").value; if (!name) return;
  try {
    const m = await api(`/api/maps/${encodeURIComponent(name)}`);
    const groups = m.groups || {};
    const pv = await api("/api/maps/preview", { method: "POST", body: { groups } });
    // Normalise globs to explicit raw names so merge/expand/remove work member by member.
    const out = {};
    for (const k of Object.keys(groups)) out[k] = groups[k];
    for (const c of pv.classes) {
      const raws = c.members.map(rawOf).filter(Boolean);
      if (out[c.name] !== undefined) out[c.name] = raws;
    }
    for (const k of Object.keys(out)) if (!out[k].length) delete out[k];
    T.groups = out; T.colors = {}; T.mapName = name; T.checked.clear();
    changed();
  } catch { /* toasted */ }
}
// Cards render together with their preview (N, shares) so the two never disagree.
function changed() { for (const g of [...T.selGroups]) if (!T.groups[g]) T.selGroups.delete(g); schedulePreview.flush(); }
const schedulePreview = debounce(async () => {
  try { T.preview = await api("/api/maps/preview", { method: "POST", body: { groups: T.groups }, quiet: true }); }
  catch (e) { toast(`Preview failed: ${e.message}`); renderGroups(); renderTable(); return; }
  renderGroups(); renderTable();
  if (active) editor.setViewPreview(T.preview);
}, 150);

function renderGroups() {
  const pv = T.preview, host = $("#tGroups");
  $("#tN").textContent = pv ? String(pv.n) : "";
  const cardFor = (name) => {
    const members = T.groups[name], isVoid = name === IGN;
    const pc = pv?.classes.find((c) => (c.name === name || (isVoid && c.void && c.name === IGN)));
    const share = pc && pv.pixel_share ? pv.pixel_share[pc.id] : undefined;
    const color = T.colors[name] || pc?.color || S.byRaw.get(members[0])?.color || [128, 128, 128];
    return el("div", { class: `gcard${isVoid ? " void" : ""}`, "data-testid": "group-card", dataset: { name } },
      el("header", {},
        !isVoid && el("input", { type: "checkbox", "data-testid": "group-select", title: "Select this group for Merge", checked: T.selGroups.has(name), onchange: (e) => { e.target.checked ? T.selGroups.add(name) : T.selGroups.delete(name); syncButtons(); } }),
        el("span", { class: "sw big", style: { background: isVoid ? "repeating-linear-gradient(45deg,#555,#555 4px,#222 4px,#222 8px)" : rgbCss(color) } }),
        el("b", {}, isVoid ? "Void (-> 255 ignore)" : name), el("span", { class: "muted mono" }, `${members.length} class${members.length > 1 ? "es" : ""}${share !== undefined ? " | " + pct(share) : ""}`),
        el("span", { class: "grow" }),
        !isVoid && el("button", { title: "Rename this group", onclick: () => renameGroup(name) }, "Rename"),
        !isVoid && el("button", { title: "Move this group under __ignore__", onclick: () => { T.groups[IGN] = [...(T.groups[IGN] || []), ...members]; delete T.groups[name]; changed(); } }, "Void"),
        el("button", { "data-testid": "group-expand", title: "Return members to their own classes", onclick: () => { delete T.groups[name]; changed(); } }, "Expand")),
      el("div", { class: "chips" }, members.map((m) => el("span", { class: "chip" }, S.byRaw.get(m)?.name ?? m,
        el("button", { title: "Remove from group", onclick: () => { removeEverywhere([m]); changed(); } }, "x")))));
  };
  const names = Object.keys(T.groups).sort((a, b) => (a === IGN) - (b === IGN));
  const un = pv?.unassigned ?? [];
  host.replaceChildren(
    ...(names.length ? names.map(cardFor) : [el("p", { class: "hint pad" }, "Identity map: every class keeps its own id. Check classes on the left and Merge to build groups.")]),
    el("div", { class: "unassigned" },
      el("h2", {}, `Unassigned (${un.length}), keep their own class`),
      el("div", { class: "chips" }, un.map((id) => el("span", { class: "chip plain" }, S.byId.get(id)?.name ?? id)))));
  $("#tName").placeholder = `HackEye<yy.major.minor>-${pv ? pv.n : "<N>"}`;
}
async function renameGroup(old) {
  const v = await dialog("Rename group", (b) => { const n = el("input", { type: "text", "data-testid": "dlg-name", value: old, title: "Group name" }); b.append(el("label", {}, "Name", n)); return () => (n.value.trim() && n.value.trim() !== IGN ? n.value.trim() : undefined); });
  if (!v || v === old) return;
  T.groups[v] = [...(T.groups[v] || []), ...T.groups[old]]; delete T.groups[old];
  if (T.colors[old]) { T.colors[v] = T.colors[old]; delete T.colors[old]; }
  changed();
}

// ---------------------------------------------------------------- save / export
function nameOk() {
  const n = $("#tName").value.trim();
  if (!n) { toast("Type a name first (e.g. HackEye26.1.1-48)", "error", 3000); $("#tName").focus(); return null; }
  if (!NAME_RE.test(n)) { toast("Name may contain letters, digits, . _ - and must start with a letter or digit"); return null; }
  return n;
}
async function saveMap(name = nameOk()) {
  if (!name) return false;
  try {
    await api("/api/maps", { method: "POST", body: { name, groups: T.groups, overwrite: false }, quiet: true });
  } catch (e) {
    if (!/exist|409|already/i.test(e.message)) { toast(`Save map: ${e.message}`); return false; }
    if (!(await confirmDialog(`Map "${name}" already exists. Overwrite it?`, "Overwrite"))) return false;
    try { await api("/api/maps", { method: "POST", body: { name, groups: T.groups, overwrite: true } }); } catch { return false; }
  }
  T.mapName = name; T.savedSig = name + JSON.stringify(T.groups); toast(`Saved map ${name}`, "ok", 2000); refreshMapList();
  return true;
}
async function exportDataset() {
  const name = nameOk(); if (!name) return;
  if (T.savedSig !== name + JSON.stringify(T.groups) && !(await saveMap(name))) return;
  const host = $("#tJob");
  try {
    const r = await api("/api/exports", { method: "POST", body: { name, map: name } });
    renderJob(host, { status: "running", progress: 0 }, "Export");
    const job = await pollJob(api, r.job, (j) => renderJob(host, j, "Export"));
    toast(job.status === "done" ? `Dataset ${name} exported` : `Export ${job.status}`, job.status === "done" ? "ok" : "error", 5000);
  } catch { /* toasted */ }
}
