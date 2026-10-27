"""Taxonomy (classes) and merge-map operations on the master dataset."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import numpy as np
import yaml

from common.taxonomy import (
    MAX_CLASS_ID, ClassDef, Taxonomy, data_yaml_dict, load_merge_map, resolve_merge_map,
    save_merge_map, save_taxonomy_json,
)

from .store import ApiError, Store

FIRST_CUSTOM_ID = 124
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise ApiError(422, f"cannot derive a slug from {name!r}")
    return s


def _color(c) -> tuple[int, int, int]:
    if not (isinstance(c, (list, tuple)) and len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)):
        raise ApiError(422, "color must be [r,g,b] with ints 0..255")
    return (int(c[0]), int(c[1]), int(c[2]))


def _text(v, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ApiError(422, f"{what} must be a non-empty string")
    return v.strip()


def write_master_yaml(store: Store) -> None:
    """Regenerates the master data yaml (names, label_mapping) exactly as pseudo_label finalize does.

    `nc` is deliberately not written: ids may have gaps (deleted custom classes) and Ultralytics rejects
    an `nc` that differs from len(names); it derives it. The previous yaml is kept as `.bak`.
    """
    tax, root = store.tax, store.root
    data = data_yaml_dict(tax.names, tax.void_ids, root=str(root))
    data["path"] = f"/datasets/{root.name}"
    data["train"], data["val"] = "images/train", "images/val"
    path = root / f"{root.name}.yaml"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    if path.exists():
        os.replace(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)
    else:
        tmp.rename(path)


def _commit_taxonomy(store: Store, classes: list[ClassDef], action: str, payload: dict) -> None:
    try:
        tax = Taxonomy(store.tax.name, store.tax.source, classes)
    except ValueError as e:
        raise ApiError(409, str(e)) from e
    save_taxonomy_json(tax, store.tax_path)
    store.tax = tax
    write_master_yaml(store)
    store.tax_log(action, payload)


def class_list(store: Store) -> dict:
    frames, pixels = store.class_totals()
    out = []
    for c in store.tax.classes:
        out.append({"id": c.id, "name": c.name, "raw_name": c.raw_name, "hier": c.hier, "color": list(c.color),
                    "void": c.void, "parent": c.parent, "custom": c.id >= FIRST_CUSTOM_ID,
                    "frames": int(frames[c.id]), "pixels": int(pixels[c.id])})
    return {"classes": out}


def _free_id(tax: Taxonomy, extra: set[int] = frozenset()) -> int:
    used = set(tax.ids) | set(extra)
    for cid in range(FIRST_CUSTOM_ID, MAX_CLASS_ID + 1):
        if cid not in used:
            return cid
    raise ApiError(409, "no free class id left (124..254)")


def _make_class(store: Store, name, hier, color, parent, raw_name, taken: set[int]) -> ClassDef:
    name, hier = _text(name, "name"), _text(hier, "hier")
    col = _color(color)
    if parent is not None:
        try:
            store.tax.by_id(int(parent))
        except (KeyError, TypeError, ValueError):
            raise ApiError(422, f"parent {parent!r} is not a class") from None
        parent = int(parent)
    raw = _text(raw_name, "raw_name") if raw_name else f"hackeye--{slug(hier)}--{slug(name)}"
    return ClassDef(id=_free_id(store.tax, taken), name=name, raw_name=raw, hier=hier, color=col, parent=parent)


def create_class(store: Store, name, hier, color, parent=None, raw_name=None) -> dict:
    with store.lock:
        c = _make_class(store, name, hier, color, parent, raw_name, set())
        _commit_taxonomy(store, [*store.tax.classes, c], "create", c.to_json())
        return c.to_json()


def split_class(store: Store, cid: int, children: list[dict]) -> dict:
    with store.lock:
        try:
            parent = store.tax.by_id(cid)
        except KeyError:
            raise ApiError(404, f"no class {cid}") from None
        if not children:
            raise ApiError(422, "children must not be empty")
        made: list[ClassDef] = []
        for ch in children:
            made.append(_make_class(store, ch.get("name"), parent.hier, ch.get("color"), cid, None, {m.id for m in made}))
        _commit_taxonomy(store, [*store.tax.classes, *made], "split", {"parent": cid, "children": [m.to_json() for m in made]})
        return {"created": [m.to_json() for m in made]}


def patch_class(store: Store, cid: int, name=None, color=None, hier=None) -> dict:
    with store.lock:
        try:
            cur = store.tax.by_id(cid)
        except KeyError:
            raise ApiError(404, f"no class {cid}") from None
        if cid < FIRST_CUSTOM_ID:
            raise ApiError(403, "base classes cannot be edited")
        new = ClassDef(cur.id, _text(name, "name") if name is not None else cur.name, cur.raw_name,
                       _text(hier, "hier") if hier is not None else cur.hier,
                       _color(color) if color is not None else cur.color, cur.void, cur.parent)
        _commit_taxonomy(store, [new if c.id == cid else c for c in store.tax.classes], "patch",
                         {"id": cid, "name": name, "color": color, "hier": hier})
        return new.to_json()


def check_deletable(store: Store, cid: int, remap_to: int | None) -> int:
    """Validates a delete request; returns the class's total pixels."""
    try:
        store.tax.by_id(cid)
    except KeyError:
        raise ApiError(404, f"no class {cid}") from None
    if cid < FIRST_CUSTOM_ID:
        raise ApiError(403, "base classes cannot be deleted")
    px = int(store.counts[:, cid].sum())
    if px and remap_to is None:
        raise ApiError(409, f"class {cid} has {px} pixels; pass remap_to=<id>")
    if remap_to is not None:
        if remap_to == cid:
            raise ApiError(422, "remap_to must differ from the class")
        if remap_to != 255 and remap_to not in store.tax.ids:
            raise ApiError(422, f"remap_to {remap_to} is not a class")
    return px


def delete_class(store: Store, cid: int) -> None:
    """Removes the class (callers have checked / emptied it). Children inherit its parent."""
    with store.lock:
        if int(store.counts[:, cid].sum()):
            raise ApiError(409, f"class {cid} still has pixels")
        cur = store.tax.by_id(cid)
        rest = [ClassDef(c.id, c.name, c.raw_name, c.hier, c.color, c.void, cur.parent if c.parent == cid else c.parent)
                for c in store.tax.classes if c.id != cid]
        _commit_taxonomy(store, rest, "delete", {"id": cid, "raw_name": cur.raw_name})


# -------------------------------------------------------------------- maps

def maps_dir(repo: Path) -> Path:
    return Path(repo) / "taxonomies"


def map_path(repo: Path, name: str) -> Path:
    if not NAME_RE.match(name):
        raise ApiError(422, "map name must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")
    return maps_dir(repo) / f"{name}.yaml"


def check_groups(groups) -> dict[str, list[str]]:
    if not isinstance(groups, dict):  # {} is the identity map: every class keeps its own id
        raise ApiError(422, "groups must be an object {name: [members]}")
    out = {}
    for k, v in groups.items():
        if not isinstance(k, str) or not k.strip():
            raise ApiError(422, "group names must be non-empty strings")
        if not isinstance(v, list) or not all(isinstance(m, str) for m in v):
            raise ApiError(422, f"group {k!r}: members must be a list of strings")
        out[k] = list(v)
    return out


def list_maps(store: Store) -> list[dict]:
    out = []
    for p in sorted(maps_dir(store.repo).glob("*.yaml")):
        try:
            n = resolve_merge_map(store.tax, load_merge_map(p)).n
        except Exception:  # noqa: BLE001 -- a map that no longer resolves is still listed
            n = None
        out.append({"name": p.stem, "n_classes": n})
    return out


def get_map(store: Store, name: str) -> dict:
    p = map_path(store.repo, name)
    if not p.exists():
        raise ApiError(404, f"no map {name!r}")
    try:
        return {"name": name, "groups": load_merge_map(p)}
    except ValueError as e:
        raise ApiError(422, str(e)) from e


def _resolve(store: Store, groups: dict[str, list[str]]):
    try:
        return resolve_merge_map(store.tax, groups)
    except ValueError as e:
        raise ApiError(422, str(e)) from e


def preview(store: Store, groups) -> dict:
    groups = check_groups(groups)
    m = _resolve(store, groups)
    raw = store.tax.raw_names
    listed = {cid for pats in groups.values() for pat in pats for cid, r in raw.items() if fnmatch.fnmatchcase(r, pat)}
    px = store.counts.sum(axis=0, dtype=np.int64)
    total = int(px.sum()) or 1
    share = {str(nid): float(px[mem].sum()) / total for nid, mem in m.members.items()}
    if m.ignored:
        share["255"] = (float(px[m.ignored].sum()) + float(px[255])) / total
    classes = [{"id": nid, "name": m.names[nid], "color": list(m.palette[nid]), "members": m.members[nid],
                "void": nid in m.void_ids} for nid in sorted(m.names)]
    return {"n": m.n, "classes": classes, "lut": [int(v) for v in m.lut], "pixel_share": share,
            "unassigned": [cid for cid in store.tax.ids if cid not in listed]}


def save_map(store: Store, name: str, groups, comment: str | None, overwrite: bool) -> dict:
    p = map_path(store.repo, name)
    groups = check_groups(groups)
    m = _resolve(store, groups)
    with store.lock:
        if p.exists() and not overwrite:
            raise ApiError(409, f"map {name!r} exists")
        p.parent.mkdir(parents=True, exist_ok=True)
        save_merge_map(groups, p, (comment or "").strip())
        store.tax_log("save-map", {"name": name, "n": m.n})
    return {"name": name, "n_classes": m.n}


