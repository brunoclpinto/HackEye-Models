"""Parses Mapillary Vistas' config_v2.0.json into the taxonomy this converter
and trainer need: dense 0..123 names, an RGB palette for overlays, and the set
of class ids that should collapse to the ignore label.

config_v2.0.json is the only source of truth for the class list. Nothing here
should hardcode a class name/id/count — if the shipped config differs from
what's documented publicly (it has shifted across Vistas releases), this
module should still produce a correct taxonomy from whatever config it's
pointed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IGNORE_LABEL = 255


@dataclass(frozen=True)
class VistasTaxonomy:
    names: dict[int, str]  # 0..n-1 -> short display name, dense, unique
    palette: dict[int, tuple[int, int, int]]  # 0..n-1 -> RGB, from the config
    void_ids: list[int]  # class ids (in the ORIGINAL 0..n-1 space) to ignore
    raw_names: dict[int, str]  # 0..n-1 -> full hierarchical name, for debugging

    @property
    def n(self) -> int:
        return len(self.names)

    def to_data_yaml_dict(self, root: str, masks_dir: str = "masks") -> dict:
        """Builds the dict Ultralytics' semantic dataset YAML expects.

        Deliberately emits `names` only, never `nc` — check_det_dataset derives
        nc = len(names) and raises if both are given and disagree, so leaving
        nc out is one less place for this to desync as the taxonomy evolves.
        """
        label_mapping: dict[int | str, int | str] = {vid: "ignore_label" for vid in self.void_ids}
        return {
            "path": root,
            "train": "images/train",
            "val": "images/val",
            "masks_dir": masks_dir,
            "names": dict(self.names),
            "label_mapping": label_mapping,
        }


def _short_name(raw_name: str) -> str:
    """'construction--flat--curb' -> 'flat curb'. Joins the last two
    hierarchy components; collisions are resolved by the caller falling back
    to the full raw_name, since a bare leaf ('curb' vs 'crosswalk-zebra' vs
    'rider') collides too easily across Vistas' hierarchy on its own.
    """
    parts = raw_name.split("--")
    return " ".join(parts[-2:]) if len(parts) >= 2 else raw_name


def load_taxonomy(config_path: str | Path) -> VistasTaxonomy:
    with open(config_path) as f:
        data = json.load(f)

    labels = data["labels"]  # list, index in list == class id (VERIFIED: Vistas' PNG pixel values are indices into this list)

    raw_names: dict[int, str] = {}
    palette: dict[int, tuple[int, int, int]] = {}
    void_ids: list[int] = []
    short_names: dict[int, str] = {}

    for cid, entry in enumerate(labels):
        raw_names[cid] = entry["name"]
        palette[cid] = tuple(entry["color"])
        if not entry.get("evaluate", True):
            void_ids.append(cid)
        short_names[cid] = _short_name(entry["name"])

    # Resolve collisions by falling back to the full hierarchical name.
    seen: dict[str, int] = {}
    final_names: dict[int, str] = {}
    collided_first: set[str] = set()
    for cid, name in short_names.items():
        if name in seen:
            collided_first.add(name)
        else:
            seen[name] = cid

    for cid, name in short_names.items():
        if name in collided_first:
            final_names[cid] = raw_names[cid]
        else:
            final_names[cid] = name

    names_list = list(final_names.values())
    if len(set(names_list)) != len(names_list):
        dupes = {n for n in names_list if names_list.count(n) > 1}
        raise ValueError(
            f"vistas_config: class names still collide after fallback to full "
            f"hierarchical name: {dupes}. config_v2.0.json may have changed shape."
        )

    return VistasTaxonomy(names=final_names, palette=palette, void_ids=void_ids, raw_names=raw_names)
