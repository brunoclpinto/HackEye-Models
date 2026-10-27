"""Applies a curation keep list to the HackEye frames dataset: DESTRUCTIVE (with --apply).
Spec: datasets/hackeye/CURATION.md. Without --apply it is a dry run that prints what would
happen and changes nothing.

Only the chosen split is trimmed (default train; val is never curated). Input is
<root>/curation/<config>/keep_<split>.csv (stem,group,reason), made by curate.py.

    <root>/images/<split>/<stem>.jpg, masks/<split>/<stem>.png   deleted when the stem is not kept
    <root>/meta/labels_<split>.csv, classpix_<split>.csv         rewritten: last row per stem, kept stems only
    <root>/meta/frames_<clip>.csv                                rows of <split> not kept dropped; other
                                                                 splits untouched; a csv left empty is deleted
    <root>/meta/curation.json                                    config, params, counts, keep-list sha256, time

Why it is built the way it is:

* Validate first, change later. Every kept stem must have a jpg, a png, a labels row, a classpix
  row and a frames row of that split, or the script aborts before touching anything.
* Order is csvs, then files, then curation.json. Each csv is rewritten atomically (tmp + os.replace),
  so a kill leaves every file either old or new. Deleting files comes last, so the csvs never
  point at a frame that is gone. A re-run after any interruption finishes the job: rewriting the
  csvs again is a no-op, and the delete step removes whatever is still on disk and not kept.
  Validation still passes on a re-run, since it only needs the kept stems.
* frames_before is the number of jpgs of the split found at the first run. A re-run on a
  half-applied dataset keeps the value of an earlier curation.json for the same config, else the
  number it finds.
* Afterwards run pseudo_label.py --finalize-only to rebuild frames.csv, the yaml and
  dataset_manifest.json (it then adds curation.json under "curation").

Run (host has no numpy, but this script needs none; use the training image for consistency):

    cd docker && docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train \\
        datasets/hackeye/apply_curation.py                 # dry run
    cd docker && docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train \\
        datasets/hackeye/apply_curation.py --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Reads a csv, ignoring a truncated final line left by a killed run."""
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header, body = rows[0], rows[1:]
    return header, [dict(zip(header, r)) for r in body if len(r) == len(header)]


def _write_csv_atomic(path: Path, header: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[c] for c in header])
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001 -- git missing in the container is normal
        return None


def _config_params(root: Path, config: str) -> dict:
    """Global sweep params + the config's own numbers from curation/report.json (if present)."""
    p: dict = {}
    m = re.fullmatch(r"fps(\d+(?:\.\d+)?)_sim(\d+(?:\.\d+)?)", config)
    if m:
        p["base_fps"], p["sim"] = float(m.group(1)), float(m.group(2))
    rp = root / "curation" / "report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        for k in ("split", "max_gap", "protect_px", "rare_frac", "hard_lowconf"):
            if k in rep.get("meta", {}):
                p[k] = rep["meta"][k]
        cfg = rep.get("configs", {}).get(config)
        if cfg:
            p["result"] = {k: v for k, v in cfg.items() if isinstance(v, (int, float))}
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/datasets/HackEye26.1-124"))
    ap.add_argument("--config", default="fps2_sim0.90")
    ap.add_argument("--keep", type=Path, default=None, help="default <root>/curation/<config>/keep_<split>.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--apply", action="store_true", help="really change the dataset (default: dry run)")
    args = ap.parse_args()

    root, split = args.root, args.split
    meta = root / "meta"
    keep_path = args.keep or root / "curation" / args.config / f"keep_{split}.csv"
    tag = "[apply]" if args.apply else "[dry-run]"
    if not keep_path.exists():
        sys.exit(f"{tag} FAILED: keep list not found: {keep_path}")

    # ---- load
    _, keep_rows = _read_csv(keep_path)
    keep_all = [r["stem"] for r in keep_rows]
    keep = set(keep_all)
    print(f"{tag} keep list {keep_path}: {len(keep_all)} rows, {len(keep)} unique stems", flush=True)
    if len(keep) != len(keep_all):
        print(f"{tag} note: {len(keep_all) - len(keep)} duplicate stems in the keep list, deduped", flush=True)
    if not keep:
        sys.exit(f"{tag} FAILED: empty keep list")

    img_dir, msk_dir = root / "images" / split, root / "masks" / split
    jpg = {p.stem for p in img_dir.glob("*.jpg")}
    png = {p.stem for p in msk_dir.glob("*.png")}
    lab_hdr, lab_rows = _read_csv(meta / f"labels_{split}.csv")
    cls_hdr, cls_rows = _read_csv(meta / f"classpix_{split}.csv")
    lab_stems = {r["stem"] for r in lab_rows}
    cls_stems = {r["stem"] for r in cls_rows}
    clip_csvs = sorted(meta.glob("frames_*.csv"))
    clip_data: dict[Path, tuple[list[str], list[dict]]] = {p: _read_csv(p) for p in clip_csvs}
    frame_stems = {r["stem"] for _, rows in clip_data.values() for r in rows if r.get("split") == split}
    print(f"{tag} {split} on disk: {len(jpg)} jpg, {len(png)} png; csv rows: {len(lab_stems)} labels, "
          f"{len(cls_stems)} classpix, {len(frame_stems)} frames in {len(clip_csvs)} clip csvs", flush=True)

    # ---- validate
    problems = []
    for name, have in (("jpg", jpg), ("png", png), ("labels row", lab_stems),
                       ("classpix row", cls_stems), ("frames row", frame_stems)):
        miss = sorted(keep - have)
        if miss:
            problems.append(f"{len(miss)} kept stems without {name}, e.g. {miss[:3]}")
    if problems:
        sys.exit(f"{tag} FAILED, nothing changed: " + "; ".join(problems))
    print(f"{tag} validation ok: all {len(keep)} kept stems have jpg, png, labels, classpix and frames rows", flush=True)

    sha = _sha256(keep_path)
    ondisk = jpg | png
    drop_files = sorted(ondisk - keep)
    frames_before = len(jpg)
    prev = meta / "curation.json"
    if prev.exists():
        try:
            old = json.loads(prev.read_text())
            if old.get("config") == args.config and old.get("split") == split and old.get("keep_sha256") == sha:
                frames_before = old["frames_before"]
        except Exception:  # noqa: BLE001 -- unreadable old file: use the number found now
            pass

    # ---- plan
    def last_per_stem(rows: list[dict]) -> list[dict]:
        d: dict[str, dict] = {}
        for r in rows:
            d[r["stem"]] = r  # last row wins
        return [r for s, r in d.items() if s in keep]

    new_lab, new_cls = last_per_stem(lab_rows), last_per_stem(cls_rows)
    print(f"{tag} labels_{split}.csv: {len(lab_rows)} -> {len(new_lab)} rows", flush=True)
    print(f"{tag} classpix_{split}.csv: {len(cls_rows)} -> {len(new_cls)} rows", flush=True)
    clip_plan: dict[Path, list[dict] | None] = {}  # None = delete the csv
    for p, (hdr, rows) in clip_data.items():
        out = [r for r in rows if r.get("split") != split or r["stem"] in keep]
        clip_plan[p] = out if out else None
        if len(out) != len(rows):
            note = "  -> csv EMPTY, would be deleted" if not out else ""
            print(f"{tag} {p.name}: {len(rows)} -> {len(out)} rows{note}", flush=True)
    n_del_csv = sum(1 for v in clip_plan.values() if v is None)
    print(f"{tag} files to delete: {len(drop_files)} stems "
          f"({len(jpg - keep)} jpg, {len(png - keep)} png); frames before {frames_before}, kept {len(keep)}", flush=True)
    if not args.apply:
        print(f"{tag} DRY RUN: nothing changed. {n_del_csv} clip csv(s) would be deleted. Re-run with --apply.", flush=True)
        return

    # ---- apply: csvs first, atomically
    _write_csv_atomic(meta / f"labels_{split}.csv", lab_hdr, new_lab)
    _write_csv_atomic(meta / f"classpix_{split}.csv", cls_hdr, new_cls)
    print(f"{tag} labels + classpix rewritten", flush=True)
    for p, out in clip_plan.items():
        if out is None:
            p.unlink()
            print(f"{tag} deleted empty {p.name} (every row was in {split} and dropped)", flush=True)
        elif len(out) != len(clip_data[p][1]):
            _write_csv_atomic(p, clip_data[p][0], out)
    print(f"{tag} frames_*.csv rewritten", flush=True)

    # ---- then delete the files of dropped stems
    n = 0
    for i, s in enumerate(drop_files, 1):
        (img_dir / f"{s}.jpg").unlink(missing_ok=True)
        (msk_dir / f"{s}.png").unlink(missing_ok=True)
        n += 1
        if i % 20000 == 0:
            print(f"{tag} deleted {i}/{len(drop_files)}", flush=True)
    left_jpg = sum(1 for _ in img_dir.glob("*.jpg"))
    left_png = sum(1 for _ in msk_dir.glob("*.png"))
    print(f"{tag} deleted {n} stems; left on disk: {left_jpg} jpg, {left_png} png", flush=True)
    if left_jpg != len(keep) or left_png != len(keep):
        sys.exit(f"{tag} FAILED: expected {len(keep)} jpg and png left, found {left_jpg}/{left_png}")

    # ---- record
    rec = {
        "config": args.config, "params": _config_params(root, args.config), "split": split,
        "frames_before": frames_before, "kept": len(keep), "deleted": frames_before - len(keep),
        "keep_list": str(keep_path), "keep_sha256": sha,
        "applied": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(Path("/app")) or _git_commit(Path(__file__).resolve().parents[2]),
    }
    _atomic_text(meta / "curation.json", json.dumps(rec, indent=1))
    print(f"{tag} wrote {meta / 'curation.json'}; next: pseudo_label.py --finalize-only", flush=True)


if __name__ == "__main__":
    main()
