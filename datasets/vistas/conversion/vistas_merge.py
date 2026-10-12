"""Derives a merged-taxonomy dataset from an already-converted one (the
output of vistas_convert.py), collapsing classes per a merge map such as
merges/v1.yaml.

Works from the converted masks, not the raw archive: a LUT remap commutes
with nearest-neighbour resizing (both are per-pixel lookups of the same
source pixel), so lut[resize(mask)] == resize(lut[mask]) exactly and there is
no reason to re-decode/re-resize 18k raw images. Images are hardlinked, not
copied -- the merged dataset costs only its masks on disk.

Every written mask is read back and compared against lut[source]. The
manifest records per-split pixel counts for both taxonomies, and the run
fails unless each merged class's count equals the sum of its members'.

    python vistas_merge.py --src /datasets/vistas-yolo --dst /datasets/vistas-yolo-merge-v1 \\
        --map datasets/vistas/conversion/merges/v1.yaml --config /datasets/vistas-raw/config_v2.0.json
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

from vistas_config import IGNORE_LABEL, build_merge_lut, load_taxonomy
from vistas_convert import _build_palette_bytes


def _merge_one(
    stem: str, src_img: Path, src_mask: Path, dst_img: Path, dst_mask: Path, lut: np.ndarray, palette_bytes: list[int]
) -> tuple[str, str | None, np.ndarray | None, np.ndarray | None]:
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
        mask_im.putpalette(palette_bytes)
        mask_im.save(dst_mask, format="PNG")

        with Image.open(dst_mask) as m:
            if not np.array_equal(np.asarray(m), out):
                return stem, "read-back mismatch", None, None
        return stem, None, np.bincount(src.ravel(), minlength=256), np.bincount(out.ravel(), minlength=256)
    except Exception as e:  # noqa: BLE001 -- one bad file must not kill an 18k-image batch
        return stem, f"{type(e).__name__}: {e}", None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="converted dataset root (output of vistas_convert.py)")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path, help="merge map YAML, e.g. merges/v1.yaml")
    ap.add_argument("--config", required=True, type=Path, help="raw Vistas config_v2.0.json")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="resolve the map and print it, write nothing")
    args = ap.parse_args()

    taxonomy = load_taxonomy(args.config)
    groups = yaml.safe_load(args.map.read_text())
    merged = build_merge_lut(taxonomy, groups)

    print(f"[vistas_merge] {taxonomy.n} -> {merged.n} classes "
          f"({len(merged.void_ids)} still void: {[merged.names[i] for i in merged.void_ids]})", file=sys.stderr)
    for new_id, name in merged.names.items():
        mem = merged.members[new_id]
        if len(mem) > 1:
            listed = ", ".join(taxonomy.raw_names[c] + ("*" if c in taxonomy.void_ids else "") for c in mem)
            print(f"  {new_id:3d} {name:<20} <- {listed}", file=sys.stderr)
    if args.dry_run:
        print("[vistas_merge] dry run, nothing written (* = void in source, un-voided)", file=sys.stderr)
        return

    with open(args.src / "vistas-v2.0.yaml") as f:
        src_yaml = yaml.safe_load(f)
    if len(src_yaml["names"]) != taxonomy.n:
        sys.exit(f"[vistas_merge] FATAL: {args.src} has {len(src_yaml['names'])} classes, expected the "
                 f"un-merged {taxonomy.n}-class conversion (was it built with --drop-void?)")

    palette_bytes = _build_palette_bytes(merged.palette, merged.n)
    manifest = {
        "source": str(args.src.resolve()),
        "map": str(args.map),
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "n_classes": merged.n,
        "void_ids": merged.void_ids,
        "members": {merged.names[i]: [taxonomy.raw_names[c] for c in m] for i, m in merged.members.items()},
        "old_to_new": {int(c): int(merged.lut[c]) for c in range(taxonomy.n)},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {},
    }

    failed = False
    for split in args.splits.split(","):
        src_imgs, src_masks = args.src / "images" / split, args.src / "masks" / split
        dst_imgs, dst_masks = args.dst / "images" / split, args.dst / "masks" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_masks.mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in src_masks.glob("*.png"))

        src_counts = np.zeros(256, np.int64)
        dst_counts = np.zeros(256, np.int64)
        errors: list[str] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_merge_one, s, src_imgs / f"{s}.jpg", src_masks / f"{s}.png",
                            dst_imgs / f"{s}.jpg", dst_masks / f"{s}.png", merged.lut, palette_bytes)
                for s in stems
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                stem, err, sc, dc = fut.result()
                if err:
                    errors.append(f"{stem}: {err}")
                else:
                    src_counts += sc
                    dst_counts += dc
                if i % 2000 == 0:
                    print(f"[vistas_merge] {split}: {i}/{len(stems)}", file=sys.stderr)

        # Pixel accounting: every merged class must hold exactly its members' pixels.
        mismatches = []
        for new_id, mem in merged.members.items():
            expected = int(src_counts[mem].sum())
            if int(dst_counts[new_id]) != expected:
                mismatches.append(f"{merged.names[new_id]}: {int(dst_counts[new_id])} != {expected}")
        stray = int(dst_counts[merged.n:IGNORE_LABEL].sum())
        if stray:
            mismatches.append(f"{stray} pixels with ids in [{merged.n}, {IGNORE_LABEL})")

        print(f"[vistas_merge] {split}: {len(stems) - len(errors)}/{len(stems)} merged, {len(errors)} errors, "
              f"pixel accounting {'OK' if not mismatches else 'FAILED'}", file=sys.stderr)
        for e in (errors + mismatches)[:30]:
            print(f"  FAIL: {e}", file=sys.stderr)
        failed |= bool(errors or mismatches)
        manifest["splits"][split] = {
            "merged": len(stems) - len(errors), "errors": errors[:50], "accounting_mismatches": mismatches,
            "pixels_old": {taxonomy.names[c]: int(src_counts[c]) for c in range(taxonomy.n)},
            "pixels_new": {merged.names[c]: int(dst_counts[c]) for c in range(merged.n)},
        }

    data_yaml = {
        "path": str(args.dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": dict(merged.names),
        "label_mapping": {vid: "ignore_label" for vid in merged.void_ids},
    }
    with open(args.dst / "vistas-v2.0.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    (args.dst / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[vistas_merge] wrote {args.dst / 'vistas-v2.0.yaml'} and merge_manifest.json", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
