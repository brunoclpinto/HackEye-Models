"""Derives a merged-taxonomy dataset from any converted/master dataset, collapsing
classes per a merge map (taxonomies/*.yaml).

Source layouts handled (read from the source data yaml):
    Vistas converted (vistas_convert.py):  images/<split>, masks/<split>, vistas-v2.0.yaml
    HackEye master:                        images/<split>, masks/<split>, <name>.yaml, taxonomy.json
The image dir of each split is the yaml's `train` / `val` entry; the mask dir is the
same path with its leading `images` component replaced by the yaml's `masks_dir`.

A LUT remap commutes with nearest-neighbour resizing, so the derived masks are exact
and no re-decode of raw data is needed. Images are hardlinked (cost: masks only).
Every written mask is read back and compared with lut[source]; the run fails unless
each merged class's pixel count equals the sum of its members' (and ignore pixels
carry over unchanged).

    python datasets/common/dataset_merge.py --src /datasets/HackEye26.1-124 \\
        --dst /datasets/HackEye26.1-48 --map taxonomies/HackEye26.1-48.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.taxonomy import (  # noqa: E402
    IGNORE_LABEL, Taxonomy, data_yaml_dict, from_vistas_config, load_merge_map, load_taxonomy_json,
    palette_bytes, resolve_merge_map,
)

TAG = "[dataset_merge]"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def _find_src_yaml(src: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    cand = src / f"{src.name}.yaml"
    if cand.exists():
        return cand
    found = sorted(src.glob("*.yaml"))
    if len(found) != 1:
        raise SystemExit(f"{TAG} FATAL: cannot pick the data yaml in {src} ({[p.name for p in found]}); pass --src-yaml")
    return found[0]


def _split_dirs(src_yaml: dict, split: str) -> tuple[str, str]:
    """(image dir, mask dir) relative to the dataset root."""
    img_rel = src_yaml[split]
    parts = Path(img_rel).parts
    masks_dir = src_yaml.get("masks_dir", "masks")
    mask_rel = str(Path(masks_dir, *parts[1:])) if parts and parts[0] == "images" else str(Path(masks_dir) / img_rel)
    return img_rel, mask_rel


def _merge_one(stem: str, src_img: Path, src_mask: Path, dst_img: Path, dst_mask: Path, lut: np.ndarray,
               pal: list[int]) -> tuple[str, str | None, np.ndarray | None, np.ndarray | None]:
    """Returns (stem, error, src_counts[256], dst_counts[256])."""
    try:
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        os.link(src_img, dst_img)

        with Image.open(src_mask) as m:
            if m.mode not in ("P", "L"):
                return stem, f"mask mode {m.mode!r}, expected P or L", None, None
            src = np.asarray(m)
        out = lut[src]
        mask_im = Image.fromarray(out, mode="P")
        mask_im.putpalette(pal)
        mask_im.save(dst_mask, format="PNG")

        with Image.open(dst_mask) as m:
            if not np.array_equal(np.asarray(m), out):
                return stem, "read-back mismatch", None, None
        return stem, None, np.bincount(src.ravel(), minlength=256), np.bincount(out.ravel(), minlength=256)
    except Exception as e:  # noqa: BLE001 -- one bad file must not kill a big batch
        return stem, f"{type(e).__name__}: {e}", None, None


def run(src: Path, dst: Path, map_path: Path, tax: Taxonomy, taxonomy_sha: str, splits: list[str], workers: int,
        dry_run: bool, src_yaml_path: Path | None = None, yaml_name: str | None = None,
        progress=None) -> bool:
    """Returns True on full success. `progress(split, done, total)` is an optional callback."""
    groups = load_merge_map(map_path)
    merged = resolve_merge_map(tax, groups)
    raw = tax.raw_names

    _log(f"{tax.n} -> {merged.n} classes ({len(merged.void_ids)} still void: "
         f"{[merged.names[i] for i in merged.void_ids]})")
    for new_id, name in merged.names.items():
        mem = merged.members[new_id]
        if len(mem) > 1:
            listed = ", ".join(raw[c] + ("*" if c in tax.void_ids else "") for c in mem)
            _log(f"  {new_id:3d} {name:<20} <- {listed}")
    if dry_run:
        _log("dry run, nothing written (* = void in source, un-voided)")
        return True

    ysrc = _find_src_yaml(src, src_yaml_path)
    src_yaml = yaml.safe_load(ysrc.read_text())
    if len(src_yaml["names"]) != tax.n:
        sys.exit(f"{TAG} FATAL: {ysrc} lists {len(src_yaml['names'])} classes, the taxonomy has {tax.n}")
    if {int(k) for k in src_yaml["names"]} != set(tax.ids):
        sys.exit(f"{TAG} FATAL: class ids in {ysrc.name} differ from the taxonomy")

    pal = palette_bytes(merged.palette, merged.n)
    manifest = {
        "source": str(src.resolve()),
        "map": str(map_path),
        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "taxonomy": tax.name,
        "taxonomy_sha256": taxonomy_sha,
        "n_classes": merged.n,
        "void_ids": merged.void_ids,
        "members": {merged.names[i]: [raw[c] for c in m] for i, m in merged.members.items()},
        "ignored": [raw[c] for c in merged.ignored],
        "old_to_new": {int(c): int(merged.lut[c]) for c in tax.ids},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {},
    }

    failed = False
    known = np.zeros(256, bool)
    known[tax.ids] = True
    for split in splits:
        img_rel, mask_rel = _split_dirs(src_yaml, split)
        src_imgs, src_masks = src / img_rel, src / mask_rel
        dst_imgs, dst_masks = dst / img_rel, dst / mask_rel
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_masks.mkdir(parents=True, exist_ok=True)
        img_by_stem = {p.stem: p for p in src_imgs.iterdir() if p.suffix.lower() in IMG_EXTS}
        stems = sorted(p.stem for p in src_masks.glob("*.png"))
        missing = [s for s in stems if s not in img_by_stem]

        src_counts = np.zeros(256, np.int64)
        dst_counts = np.zeros(256, np.int64)
        errors: list[str] = [f"{s}: no image" for s in missing]
        todo = [s for s in stems if s in img_by_stem]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_merge_one, s, img_by_stem[s], src_masks / f"{s}.png",
                            dst_imgs / img_by_stem[s].name, dst_masks / f"{s}.png", merged.lut, pal)
                for s in todo
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                stem, err, sc, dc = fut.result()
                if err:
                    errors.append(f"{stem}: {err}")
                else:
                    src_counts += sc
                    dst_counts += dc
                if progress:
                    progress(split, i, len(todo))
                if i % 2000 == 0:
                    _log(f"{split}: {i}/{len(todo)}")

        # Pixel accounting: every merged class must hold exactly its members' pixels.
        mismatches = []
        for new_id, mem in merged.members.items():
            expected = int(src_counts[mem].sum())
            if int(dst_counts[new_id]) != expected:
                mismatches.append(f"{merged.names[new_id]}: {int(dst_counts[new_id])} != {expected}")
        unknown = int(src_counts[~known & (np.arange(256) != IGNORE_LABEL)].sum())
        if unknown:
            mismatches.append(f"{unknown} source pixels with ids not in the taxonomy")
        want_ign = int(src_counts[IGNORE_LABEL]) + unknown + int(src_counts[merged.ignored].sum())
        if int(dst_counts[IGNORE_LABEL]) != want_ign:
            mismatches.append(f"ignore pixels {int(dst_counts[IGNORE_LABEL])} != {want_ign}")
        stray = int(dst_counts[merged.n:IGNORE_LABEL].sum())
        if stray:
            mismatches.append(f"{stray} pixels with ids in [{merged.n}, {IGNORE_LABEL})")

        _log(f"{split}: {len(todo) - len([e for e in errors if not e.endswith('no image')])}/{len(stems)} merged, "
             f"{len(errors)} errors, pixel accounting {'OK' if not mismatches else 'FAILED'}")
        for e in (errors + mismatches)[:30]:
            _log(f"  FAIL: {e}")
        failed |= bool(errors or mismatches)
        manifest["splits"][split] = {
            "merged": len(todo) - len([e for e in errors if not e.endswith("no image")]),
            "errors": errors[:50], "accounting_mismatches": mismatches,
            "pixels_old": {tax.names[c]: int(src_counts[c]) for c in tax.ids},
            "pixels_new": {merged.names[c]: int(dst_counts[c]) for c in range(merged.n)},
        }

    data = data_yaml_dict(merged.names, merged.void_ids, str(dst.resolve()), src_yaml.get("masks_dir", "masks"))
    data["train"], data["val"] = src_yaml["train"], src_yaml["val"]
    out_yaml = dst / (yaml_name or f"{dst.name}.yaml")
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (dst / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    _log(f"wrote {out_yaml} and merge_manifest.json")
    return not failed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="source dataset root")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path, help="merge map YAML, e.g. taxonomies/HackEye26.1-48.yaml")
    ap.add_argument("--taxonomy", type=Path, default=None,
                    help="source taxonomy json (default <src>/taxonomy.json; a Vistas config_v2.0.json is also accepted)")
    ap.add_argument("--src-yaml", type=Path, default=None, help="source data yaml (default <src>/<src.name>.yaml or the only *.yaml)")
    ap.add_argument("--yaml-name", default=None, help="output yaml file name (default <dst.name>.yaml)")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="resolve the map and print it, write nothing")
    args = ap.parse_args(argv)

    tpath = args.taxonomy or (args.src / "taxonomy.json")
    if not tpath.exists():
        sys.exit(f"{TAG} FATAL: no taxonomy at {tpath}; pass --taxonomy")
    if tpath.name.startswith("config_"):
        tax = from_vistas_config(tpath)
    else:
        tax = load_taxonomy_json(tpath)
    ok = run(args.src, args.dst, args.map, tax, hashlib.sha256(tpath.read_bytes()).hexdigest(),
             args.splits.split(","), args.workers, args.dry_run, args.src_yaml, args.yaml_name)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
