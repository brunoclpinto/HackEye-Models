"""Base taxonomies (class lists) and merge maps, shared by every dataset tool.

A *taxonomy* is the list of classes a dataset's masks are stored in: the 124
Mapillary Vistas v2.0 classes (ids 0..123), plus any custom classes appended
after them (ids up to 254; 255 is the ignore label). It travels with a dataset
as `<root>/taxonomy.json`; `taxonomies/vistas124.json` is the pristine base.

A *merge map* (e.g. `taxonomies/HackEye26.1-48.yaml`) is a non-destructive
view: merged name -> member raw names (fnmatch globs). `resolve_merge_map`
turns taxonomy + map into the LUT that derives a merged dataset. Semantics are
identical to the original `vistas_config.build_merge_lut`:

* members are matched with fnmatch on raw names; a pattern matching nothing,
  or a class claimed by two groups, is an error;
* a group's colour is its FIRST listed member's (first in listing order);
* void members are un-voided into their group;
* classes not listed keep their own class (unlisted void ones stay void, i.e.
  still mapped to ignore through the yaml's label_mapping);
* new ids are dense, ordered by the lowest original id of each output class;
* the reserved group `__ignore__` (IGNORE_GROUP) lists classes whose pixels become
  the ignore label 255 in the derived dataset; they get no output class.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

IGNORE_LABEL = 255
IGNORE_GROUP = "__ignore__"
MAX_CLASS_ID = 254

_VISTAS_CONVERSION = Path(__file__).resolve().parents[1] / "vistas" / "conversion"


@dataclass
class ClassDef:
    id: int
    name: str
    raw_name: str
    hier: str
    color: tuple[int, int, int]
    void: bool = False
    parent: int | None = None  # id of the class this one was split from

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "raw_name": self.raw_name, "hier": self.hier,
                "color": list(self.color), "void": self.void, "parent": self.parent}

    @staticmethod
    def from_json(d: dict) -> "ClassDef":
        return ClassDef(id=int(d["id"]), name=d["name"], raw_name=d["raw_name"],
                        hier=d.get("hier") or d["raw_name"].split("--")[0],
                        color=tuple(int(c) for c in d["color"]), void=bool(d.get("void", False)),
                        parent=None if d.get("parent") is None else int(d["parent"]))


@dataclass
class Taxonomy:
    name: str
    source: str
    classes: list[ClassDef] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.classes = sorted(self.classes, key=lambda c: c.id)
        ids = [c.id for c in self.classes]
        if len(set(ids)) != len(ids):
            raise ValueError("taxonomy: duplicate class ids")
        if ids and (min(ids) < 0 or max(ids) > MAX_CLASS_ID):
            raise ValueError(f"taxonomy: class ids must be in 0..{MAX_CLASS_ID} (255 = ignore)")
        for attr in ("name", "raw_name"):
            vals = [getattr(c, attr) for c in self.classes]
            if len(set(vals)) != len(vals):
                raise ValueError(f"taxonomy: duplicate class {attr}s: {sorted({v for v in vals if vals.count(v) > 1})}")

    @property
    def n(self) -> int:
        return len(self.classes)

    @property
    def max_id(self) -> int:
        return self.classes[-1].id if self.classes else -1

    @property
    def ids(self) -> list[int]:
        return [c.id for c in self.classes]

    @property
    def names(self) -> dict[int, str]:
        return {c.id: c.name for c in self.classes}

    @property
    def raw_names(self) -> dict[int, str]:
        return {c.id: c.raw_name for c in self.classes}

    @property
    def palette(self) -> dict[int, tuple[int, int, int]]:
        return {c.id: c.color for c in self.classes}

    @property
    def void_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.void]

    def by_id(self, cid: int) -> ClassDef:
        for c in self.classes:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "classes": [c.to_json() for c in self.classes]}

    @staticmethod
    def from_json(d: dict) -> "Taxonomy":
        return Taxonomy(name=d["name"], source=d.get("source", ""), classes=[ClassDef.from_json(c) for c in d["classes"]])

    def palette_bytes(self) -> list[int]:
        return palette_bytes(self.palette)


def palette_bytes(palette: dict[int, tuple[int, int, int]], n: int | None = None) -> list[int]:
    """Flat 768-entry PNG palette; ids missing from `palette` (gaps, 255) are black."""
    n = (max(palette) + 1) if n is None and palette else (n or 0)
    flat: list[int] = []
    for cid in range(256):
        flat.extend(palette[cid] if cid < n and cid in palette else (0, 0, 0))
    return flat


# ------------------------------------------------------------------ json io

def _atomic_write(path: Path, text: str, backup: bool) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    if backup and path.exists():
        os.replace(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def load_taxonomy_json(path: str | Path) -> Taxonomy:
    return Taxonomy.from_json(json.loads(Path(path).read_text()))


def save_taxonomy_json(tax: Taxonomy, path: str | Path) -> None:
    """Atomic write; the previous file (if any) is kept as `<path>.bak`."""
    _atomic_write(Path(path), json.dumps(tax.to_json(), indent=1) + "\n", backup=True)


def from_vistas_config(config_path: str | Path, name: str = "vistas124") -> Taxonomy:
    """Builds the base taxonomy from Vistas' config_v2.0.json (names as vistas_config.load_taxonomy)."""
    if str(_VISTAS_CONVERSION) not in sys.path:
        sys.path.insert(0, str(_VISTAS_CONVERSION))
    from vistas_config import load_taxonomy  # noqa: PLC0415

    vt = load_taxonomy(config_path)
    return from_vistas_taxonomy(vt, name=name)


def from_vistas_taxonomy(vt, name: str = "vistas124") -> Taxonomy:
    classes = [ClassDef(id=cid, name=vt.names[cid], raw_name=vt.raw_names[cid],
                        hier=vt.raw_names[cid].split("--")[0], color=tuple(vt.palette[cid]),
                        void=cid in vt.void_ids, parent=None)
               for cid in sorted(vt.names)]
    return Taxonomy(name=name, source="Mapillary Vistas v2.0 config_v2.0.json", classes=classes)


# ---------------------------------------------------------------- merge map

@dataclass(frozen=True)
class Merged:
    lut: np.ndarray  # 256-entry uint8: base id -> new id (255 stays 255, unknown ids -> 255)
    names: dict[int, str]
    palette: dict[int, tuple[int, int, int]]  # group's FIRST listed member's colour
    void_ids: list[int]  # new ids that remain ignored
    members: dict[int, list[int]]  # new id -> base ids
    ignored: list[int] = field(default_factory=list)  # base ids sent to 255 by the __ignore__ group

    @property
    def n(self) -> int:
        return len(self.names)


def resolve_merge_map(tax: Taxonomy, groups: dict[str, list[str]]) -> Merged:
    raw_names = tax.raw_names
    owner: dict[int, str] = {}
    resolved: dict[str, list[int]] = {}
    for group, patterns in groups.items():
        ids: list[int] = []
        for pat in patterns:
            hits = [cid for cid, raw in raw_names.items() if fnmatch.fnmatchcase(raw, pat)]
            if not hits:
                raise ValueError(f"merge group {group!r}: pattern {pat!r} matches no class")
            for cid in hits:
                if cid in owner and owner[cid] != group:
                    raise ValueError(f"class {raw_names[cid]!r} claimed by both {owner[cid]!r} and {group!r}")
                if cid not in ids:
                    ids.append(cid)
                owner[cid] = group
        resolved[group] = ids

    names0, palette0, void0 = tax.names, tax.palette, set(tax.void_ids)
    entries: list[tuple[int, str, list[int]]] = []
    ignored = sorted(resolved.get(IGNORE_GROUP, []))
    for cid in tax.ids:
        if cid not in owner:
            entries.append((cid, names0[cid], [cid]))
    for group, ids in resolved.items():
        if group == IGNORE_GROUP:
            continue
        entries.append((min(ids), group, ids))
    entries.sort(key=lambda e: e[0])

    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    names: dict[int, str] = {}
    palette: dict[int, tuple[int, int, int]] = {}
    members: dict[int, list[int]] = {}
    void_ids: list[int] = []
    for new_id, (_, name, ids) in enumerate(entries):
        for cid in ids:
            lut[cid] = new_id
        names[new_id] = name
        palette[new_id] = palette0[ids[0]]
        members[new_id] = sorted(ids)
        if len(ids) == 1 and ids[0] in void0 and ids[0] not in owner:
            void_ids.append(new_id)

    if len(set(names.values())) != len(names):
        dupes = {n for n in names.values() if list(names.values()).count(n) > 1}
        raise ValueError(f"merged class names collide: {dupes} -- rename the merge group(s)")
    return Merged(lut=lut, names=names, palette=palette, void_ids=void_ids, members=members, ignored=ignored)


def load_merge_map(path: str | Path) -> dict[str, list[str]]:
    groups = yaml.safe_load(Path(path).read_text())
    if not isinstance(groups, dict) or not all(isinstance(v, list) for v in groups.values()):
        raise ValueError(f"{path}: merge map must be a mapping of group name -> list of class names")
    return {str(k): [str(m) for m in v] for k, v in groups.items()}


class _IndentedDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):  # list items indented under their key
        return super().increase_indent(flow, False)


def save_merge_map(groups: dict[str, list[str]], path: str | Path, header_comment: str = "") -> None:
    """Writes the HackEye26.1-48.yaml format: optional '# ' header, then `group:\\n  - member`."""
    header = "".join(f"# {ln}".rstrip() + "\n" for ln in header_comment.splitlines())
    body = yaml.dump({k: list(v) for k, v in groups.items()}, Dumper=_IndentedDumper, sort_keys=False,
                     default_flow_style=False, allow_unicode=True, width=10_000)
    _atomic_write(Path(path), header + ("\n" if header else "") + body, backup=False)


def data_yaml_dict(names: dict[int, str], void_ids: list[int], root: str, masks_dir: str = "masks") -> dict:
    """Ultralytics semantic data yaml. `names` only (no `nc`: check_det_dataset derives it)."""
    return {
        "path": root,
        "train": "images/train",
        "val": "images/val",
        "masks_dir": masks_dir,
        "names": dict(names),
        "label_mapping": {vid: "ignore_label" for vid in void_ids},
    }
