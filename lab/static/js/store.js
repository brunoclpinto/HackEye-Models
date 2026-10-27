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
