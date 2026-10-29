// Shared app state: taxonomy (from the API), meta, and a tiny event bus.
import { api } from "./api.js";

export const S = { tax: [], byId: new Map(), byRaw: new Map(), meta: null, bus: new EventTarget() };
export const emit = (name, detail) => S.bus.dispatchEvent(new CustomEvent(name, { detail }));
export const on = (name, fn) => S.bus.addEventListener(name, (e) => fn(e.detail));

export async function loadTaxonomy() {
  const r = await api("/api/taxonomy");
  S.tax = r.classes;
  S.byId = new Map(S.tax.map((c) => [c.id, c]));
  S.byRaw = new Map(S.tax.map((c) => [c.raw_name, c]));
  emit("taxonomy", S.tax);
  return S.tax;
}

export async function loadMeta() {
  try { S.meta = await api("/api/meta", { quiet: true }); } catch { /* keep old */ }
  emit("meta", S.meta);
  return S.meta;
}

export const clsName = (id) => (id === 255 ? "ignore" : S.byId.get(id)?.name ?? `#${id}`);
export const clsColor = (id) => S.byId.get(id)?.color ?? (id === 255 ? [90, 90, 90] : [128, 128, 128]);
export const samAvailable = () => !!S.meta?.sam?.available;

// Display-only per-viewer colour overrides (localStorage lab.viewColors = {id: [r,g,b]}); the taxonomy itself is never changed.
export const VC = {};
try { Object.assign(VC, JSON.parse(localStorage.getItem("lab.viewColors") || "{}")); } catch { /* ignore */ }
const saveVC = () => { try { localStorage.setItem("lab.viewColors", JSON.stringify(VC)); } catch { /* ignore */ } };
export const viewColor = (id) => VC[id] ?? clsColor(id);
export const hasViewColor = (id) => !!VC[id];
export function setViewColor(id, c) { if (c) VC[id] = c; else delete VC[id]; saveVC(); emit("viewcolors", id); }
export function resetViewColors() { for (const k of Object.keys(VC)) delete VC[k]; saveVC(); emit("viewcolors", null); }
