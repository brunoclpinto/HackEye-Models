"""Writes the Models Lab's edited class-pixel counts back into the master dataset's meta csvs.

The Lab keeps the counts of every saved frame in `<root>/lab/lab.db` (table `classpix`), which
overrides `meta/classpix_<split>.csv`. This script folds those rows back into the csvs (atomic
rewrite; rows of edited stems are replaced, new stems appended) and then re-runs
`pseudo_label.py --finalize-only --dst <root>`, which rebuilds class_pixels.json, frames.csv,
the yaml and dataset_manifest.json from the csvs.

Finalize needs the dataset-creation weights only to record their sha256 in the manifest, so
`--weights` must point at an existing file (default as in pseudo_label.py, resolved against the
working directory: run it from /app in the train container). pseudo_label imports torch at module
level, hence run this in the train image:

    cd docker && docker compose run --rm --entrypoint python train \\
        datasets/hackeye/sync_lab_edits.py --root /datasets/HackEye26.1-124

Use `--no-finalize` to only rewrite the csvs. `--dry-run` reports what would change.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

SPLITS = ("train", "val")
DEFAULT_WEIGHTS = "models/Yolo26x-sem-Vistas124-HackEye-1280.pt"


def _split_by_stem(root: Path) -> dict[str, str]:
    with open(root / "frames.csv", newline="") as f:
        return {r["stem"]: r["split"] for r in csv.DictReader(f)}


def sync(root: Path, dry_run: bool = False) -> dict[str, int]:
    """Rewrites meta/classpix_<split>.csv from the db. Returns {split: rows replaced or added}."""
    db_path = root / "lab" / "lab.db"
    if not db_path.exists():
        print(f"[sync] no {db_path}: nothing to sync")
        return {s: 0 for s in SPLITS}
    split_of = _split_by_stem(root)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        edits = {stem: counts for stem, counts in con.execute("SELECT stem, counts FROM classpix")}
    finally:
        con.close()
    unknown = [s for s in edits if s not in split_of]
    if unknown:
        print(f"[sync] warning: {len(unknown)} db stems not in frames.csv (skipped), e.g. {unknown[:3]}")
    changed = {s: 0 for s in SPLITS}
    for split in SPLITS:
        mine = {st: c for st, c in edits.items() if split_of.get(st) == split}
        path = root / "meta" / f"classpix_{split}.csv"
        rows: list[list[str]] = []
        header = ["stem", "counts"]
        if path.exists():
            with open(path, newline="") as f:
                all_rows = list(csv.reader(f))
            if all_rows:
                header, rows = all_rows[0], all_rows[1:]
        seen: set[str] = set()
        out: list[list[str]] = []
        for r in rows:
            if len(r) < 2:
                continue  # truncated line from a killed labelling run
            if r[0] in mine:
                if r[0] in seen:
                    continue  # duplicate of an edited stem
                seen.add(r[0])
                if r[1] != mine[r[0]]:
                    changed[split] += 1
                out.append([r[0], mine[r[0]]])
            else:
                out.append(r)
        for st, c in mine.items():
            if st not in seen:
                out.append([st, c])
                changed[split] += 1
        print(f"[sync] {split}: {len(mine)} edited frames, {changed[split]} rows changed/added")
        if dry_run:
            continue
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(out)
        os.replace(tmp, path)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="master dataset root, e.g. /datasets/HackEye26.1-124")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="existing weights file (sha256 goes into the manifest)")
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sync(args.root, args.dry_run)
    if args.no_finalize or args.dry_run:
        return
    cmd = [sys.executable, str(Path(__file__).with_name("pseudo_label.py")), "--finalize-only",
           "--dst", str(args.root), "--weights", args.weights]
    print("[sync] " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
