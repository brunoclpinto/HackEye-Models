// Rules tab: bulk relabel from -> to across filtered frames, dry run then apply.
import { api } from "./api.js";
import { $, el, toast, confirmDialog, pollJob, renderJob, fmtPx } from "./util.js";
import { S, on, clsName } from "./store.js";
import { hashFor } from "./nav.js";

let last = null, ready = false;

export function init() {
  on("taxonomy", fillClasses);
  $("#rDry").addEventListener("click", dry);
  $("#rApply").addEventListener("click", apply);
  for (const id of ["#rFrom", "#rTo", "#rSplit", "#rClip", "#rStems"]) $(id).addEventListener("input", () => { last = null; $("#rApply").disabled = true; });
  $("#rSplit").addEventListener("change", fillClips);
}
export async function show() {
  fillClasses();
  if (!ready) { ready = true; try { S.clips = await api("/api/clips", { quiet: true }); } catch { S.clips = []; } fillClips(); }
}
function fillClasses() {
  const opts = () => [...S.tax].sort((a, b) => a.name.localeCompare(b.name)).map((c) => el("option", { value: c.id }, `${c.name} (${c.id})`));
  for (const id of ["#rFrom", "#rTo"]) { const s = $(id), cur = s.value; s.replaceChildren(...opts()); if (cur) s.value = cur; }
}
function fillClips() {
  const s = $("#rClip"), split = $("#rSplit").value;
  s.replaceChildren(el("option", { value: "" }, "all"), ...(S.clips || []).filter((c) => !split || c.split === split).map((c) => el("option", { value: c.clip }, `${c.clip} (${c.frames})`)));
}
function body() {
  const filter = {};
  if ($("#rSplit").value) filter.split = $("#rSplit").value;
  if ($("#rClip").value) filter.clip = $("#rClip").value;
  const stems = $("#rStems").value.split(/[\s,]+/).filter(Boolean);
  if (stems.length) filter.stems = stems;
  return { from: Number($("#rFrom").value), to: Number($("#rTo").value), filter };
}
async function dry() {
  const b = body();
  if (b.from === b.to) { toast("From and To are the same class"); return; }
  try {
    const r = await api("/api/rules/dry-run", { method: "POST", body: b });
    last = b;
    $("#rApply").disabled = r.frames === 0;
    $("#rResult").replaceChildren(el("div", { "data-testid": "rule-result", dataset: { frames: r.frames, pixels: r.pixels } },
      el("p", {}, el("b", {}, `${r.frames} frames, ${fmtPx(r.pixels)} px`), ` would change from ${clsName(b.from)} to ${clsName(b.to)}.`),
      el("div", { class: "thumbs" }, (r.sample || []).slice(0, 20).map((s) => el("img", { src: `/api/frames/${encodeURIComponent(s)}/thumb.jpg`, title: s, loading: "lazy", onclick: () => { location.hash = hashFor(s); } })))));
  } catch { /* toasted */ }
}
async function apply() {
  if (!last) return;
  const b = body();
  if (JSON.stringify(b) !== JSON.stringify(last)) { toast("Parameters changed: run Dry run again"); return; }
  if (!(await confirmDialog(`Relabel ${clsName(b.from)} -> ${clsName(b.to)} in the matching frames? Backups of the originals are kept.`, "Apply"))) return;
  try {
    const r = await api("/api/rules/apply", { method: "POST", body: b });
    const host = $("#rJob");
    renderJob(host, { status: "running", progress: 0 }, "Rule");
    const job = await pollJob(api, r.job, (j) => renderJob(host, j, "Rule"));
    if (job.status === "done") { toast("Rule applied", "ok", 3000); document.dispatchEvent(new CustomEvent("frames-changed")); }
    $("#rApply").disabled = true; last = null;
  } catch { /* toasted */ }
}
