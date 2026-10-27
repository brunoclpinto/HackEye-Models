"""Dry-run curation report for the HackEye frames dataset: near-duplicate trimming.
Spec: datasets/hackeye/CURATION.md. CPU only; writes ONLY under <root>/curation/,
never touches frames, masks, meta or csvs of the dataset (nothing is deleted or moved).

    <root>/curation/grid_cache_<split>.npz        cached 18x32 layout grids (48-class, uint8)
    <root>/curation/<config>/keep_<split>.csv     stem,group,reason
    <root>/curation/<config>/groups.csv           one row per layout group
    <root>/curation/<config>/hard_review.csv      least confident dropped frame per group (if unsure)
    <root>/curation/report.json, report.md        the sweep

Method (per config = base_fps x sim):
  A. per clip, bin the 20 fps index k into bins of 20/base_fps frames, keep the most confident frame per bin;
  B. walk those frames in time order, grouping by agreement of the 32x18 majority-class grid (after the
     124 -> 48 HackEye26.1-48 merge) with the group's first frame (anchor), within max_gap seconds; keep the
     most confident frame of each group (highest conf_mean, ties: lower lowconf_frac, then earlier k);
  C. protect rare / navigation classes: a group that has such a class (>= protect_px pixels in some member)
     that its kept frame lacks also keeps its best member for that class;
  D. the least confident dropped member of each group goes to hard_review.csv if lowconf_frac >= hard_lowconf.

Only frames that have a mask AND a labels row AND a classpix row are considered, so the script works on a
partially labelled dataset; re-runs only compute grids for masks not yet in the cache. Val is never trimmed
(counts only).

Run (host has no numpy, use the training image; leave the GPU alone):

    cd docker && docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train \\
        datasets/hackeye/curate.py --root /datasets/HackEye --split train
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vistas" / "conversion"))
from vistas_config import build_merge_lut, load_taxonomy  # noqa: E402

MERGE_YAML = Path(__file__).resolve().parents[1] / "vistas" / "conversion" / "merges" / "HackEye26.1-48.yaml"
W, H, BLOCK = 1280, 720, 40
GH, GW = H // BLOCK, W // BLOCK  # 18 x 32
CELLS = GH * GW  # 576
K = 256  # merged ids are < 256 (255 = ignore)
NAV_IDS = [4, 13, 14, 15, 19, 20, 22, 23, 25, 45, 56, 51, 52, 69, 74, 77, 91]  # Vistas ids, CURATION.md
HIGHLIGHT = ("fps4_sim0.90", "fps2_sim0.90")  # configs shown in the per-class tables
FPS = 20

_LUT: np.ndarray | None = None
_MASKS: Path | None = None


# ---------------------------------------------------------------- grids
def _init_worker(lut: np.ndarray, masks_dir: str) -> None:
    global _LUT, _MASKS
    _LUT, _MASKS = lut, Path(masks_dir)


def grid_of(mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """(720,1280) Vistas ids -> (18,32) uint8 majority merged class per 40x40 block (ties: lowest id)."""
    m = lut[mask]  # map first, then majority
    blocks = m.reshape(GH, BLOCK, GW, BLOCK).transpose(0, 2, 1, 3).reshape(CELLS, BLOCK * BLOCK)
    idx = blocks.astype(np.int32) + (np.arange(CELLS, dtype=np.int32) * K)[:, None]
    hist = np.bincount(idx.ravel(), minlength=CELLS * K).reshape(CELLS, K)
    return hist.argmax(1).astype(np.uint8).reshape(GH, GW)


def _grid_job(stem: str) -> tuple[str, np.ndarray]:
    mask = np.array(Image.open(_MASKS / f"{stem}.png"))
    if mask.shape != (H, W):
        raise ValueError(f"{stem}: mask shape {mask.shape}, expected {(H, W)}")
    return stem, grid_of(mask, _LUT)


def load_grids(cache: Path, stems: list[str], lut: np.ndarray, masks_dir: Path, workers: int) -> np.ndarray:
    """Returns (len(stems), 576) grids, computing only the ones missing from the cache."""
    lut_sha = hashlib.sha256(lut.tobytes()).hexdigest()
    c_stems, c_grids = np.array([], dtype="<U48"), np.zeros((0, GH, GW), np.uint8)
    if cache.exists():
        z = np.load(cache)
        if str(z["lut_sha"]) == lut_sha:
            c_stems, c_grids = z["stems"], z["grids"]
        else:
            print("[curate] grid cache was built with a different merge LUT, recomputing", flush=True)
    have = dict(zip(c_stems.tolist(), range(len(c_stems))))
    new = [s for s in stems if s not in have]
    print(f"[curate] grids: {len(stems) - len(new)} cached, {len(new)} to compute ({workers} workers)", flush=True)
    if new:
        t0, done, out = time.time(), 0, {}
        with Pool(workers, initializer=_init_worker, initargs=(lut, str(masks_dir))) as pool:
            for stem, g in pool.imap_unordered(_grid_job, new, chunksize=32):
                out[stem] = g
                done += 1
                if done % 5000 == 0 or done == len(new):
                    el = time.time() - t0
                    print(f"[curate]   {done}/{len(new)}  {done / el:.0f} grids/s  eta {(len(new) - done) / (done / el):.0f}s",
                          flush=True)
        all_stems = np.array(sorted(set(have) | set(out)), dtype="<U48")
        grids = np.zeros((len(all_stems), GH, GW), np.uint8)
        for i, s in enumerate(all_stems.tolist()):
            grids[i] = out[s] if s in out else c_grids[have[s]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(tmp, stems=all_stems, grids=grids, lut_sha=lut_sha)
        os.replace(tmp, cache)
        c_stems, c_grids = all_stems, grids
        have = dict(zip(c_stems.tolist(), range(len(c_stems))))
    return c_grids[[have[s] for s in stems]].reshape(len(stems), CELLS)


# ---------------------------------------------------------------- dataset tables
def read_csv_last_wins(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("stem") and None not in r.values():  # skip truncated trailing lines
                rows[r["stem"]] = r
    return rows


def load_frames(root: Path, split: str) -> dict:
    """Frames of `split` having mask + labels row + classpix row, sorted by (clip, k)."""
    meta = root / "meta"
    labels = read_csv_last_wins(meta / f"labels_{split}.csv")
    classpix = read_csv_last_wins(meta / f"classpix_{split}.csv")
    masks = {p.name[:-4] for p in os.scandir(root / "masks" / split) if p.name.endswith(".png")}
    recs, n_meta = [], 0
    for fcsv in sorted(meta.glob("frames_*.csv")):
        for r in read_csv_last_wins(fcsv).values():
            if r["split"] != split:
                continue
            n_meta += 1
            s = r["stem"]
            if s in masks and s in labels and s in classpix:
                recs.append((r["clip"], int(r["k"]), s, float(r["t_sec"]),
                             float(labels[s]["conf_mean"]), float(labels[s]["lowconf_frac"]), classpix[s]["counts"]))
    recs.sort(key=lambda x: (x[0], x[1]))
    return {
        "n_total": n_meta, "n_masks": len(masks), "n_labels": len(labels), "n_classpix": len(classpix),
        "stems": [r[2] for r in recs], "clip": [r[0] for r in recs],
        "k": np.array([r[1] for r in recs], np.int64), "t": np.array([r[3] for r in recs]),
        "conf": np.array([r[4] for r in recs]), "low": np.array([r[5] for r in recs]),
        "counts": [r[6] for r in recs],
    }


def class_matrix(counts: list[str]) -> np.ndarray:
    """Sparse 'id:px;id:px' strings -> (N,256) int32 pixel counts (ids are Vistas 0..123)."""
    m = np.zeros((len(counts), K), np.int32)
    for i, s in enumerate(counts):
        for kv in s.split(";") if s else ():
            c, p = kv.split(":")
            m[i, int(c)] = int(p)
    return m


# ---------------------------------------------------------------- curation
def best_key(f: dict, i: int) -> tuple:
    """Sort key, smaller = better: highest conf_mean, then lower lowconf_frac, then earlier k."""
    return (-f["conf"][i], f["low"][i], f["k"][i])


def stage_a(f: dict, clip_idx: np.ndarray, base_fps: int) -> list[int]:
    """Most confident frame per bin of 20/base_fps consecutive k, returned in time order."""
    step = FPS // base_fps
    best: dict[int, int] = {}
    for i in clip_idx:
        b = int(f["k"][i]) // step
        if b not in best or best_key(f, i) < best_key(f, best[b]):
            best[b] = int(i)
    return [best[b] for b in sorted(best)]


def curate(f: dict, grids: np.ndarray, present: np.ndarray, protected: list[int], names124: dict[int, str],
           base_fps: int, sim: float, max_gap: float, hard_lowconf: float) -> dict:
    need = math.ceil(sim * CELLS - 1e-9)  # cells that must agree
    by_clip: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(f["clip"]):
        by_clip[c].append(i)
    keep: list[tuple[int, int, str]] = []  # (frame, group id, reason)
    groups, hard = [], []
    n_a = 0
    gid = 0
    for clip, idx in by_clip.items():
        frames = stage_a(f, np.array(idx), base_fps)
        n_a += len(frames)
        cur: list[int] = []
        agree: list[int] = []

        def close() -> None:
            nonlocal gid
            if not cur:
                return
            kept = min(cur, key=lambda i: best_key(f, i))
            chosen = {kept: "best"}
            if len(cur) > 1 and protected:
                pres = present[cur][:, protected]
                missing = pres.any(0) & ~present[kept, protected]
                extra: dict[int, list[str]] = {}
                for j in np.nonzero(missing)[0]:
                    col = protected[j]
                    if any(present[e, col] for e in extra):
                        extra[next(e for e in extra if present[e, col])].append(names124[col])
                        continue
                    cand = [m for m in cur if present[m, col]]
                    e = min(cand, key=lambda i: best_key(f, i))
                    extra.setdefault(e, []).append(names124[col])
                for e, cls in extra.items():
                    chosen[e] = "protect:" + ",".join(cls)
            for i, why in chosen.items():
                keep.append((i, gid, why))
            dropped = [i for i in cur if i not in chosen]
            if dropped:
                h = min(dropped, key=lambda i: (f["conf"][i], -f["low"][i]))
                if f["low"][h] >= hard_lowconf:
                    hard.append((h, gid))
            groups.append((gid, clip, cur[0], kept, list(cur), min(agree) if agree else CELLS))
            gid += 1

        for i in frames:
            if cur:
                a = int((grids[i] == grids[cur[0]]).sum())
                if a >= need and f["t"][i] - f["t"][cur[0]] <= max_gap:
                    cur.append(i)
                    agree.append(a)
                    continue
                close()
            cur, agree = [i], []
        close()
        cur = []
    return {"keep": keep, "groups": groups, "hard": hard, "n_stage_a": n_a}


def write_config(outdir: Path, split: str, f: dict, res: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    st, t, conf = f["stems"], f["t"], f["conf"]
    with open(outdir / f"keep_{split}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "group", "reason"])
        for i, g, why in sorted(res["keep"], key=lambda x: st[x[0]]):
            w.writerow([st[i], g, why])
    with open(outdir / "groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "clip", "anchor", "kept", "n_members", "t_first", "t_last", "min_agree_cells", "members"])
        for g, clip, anchor, kept, mem, mn in res["groups"]:
            w.writerow([g, clip, st[anchor], st[kept], len(mem), f"{t[mem[0]]:.2f}", f"{t[mem[-1]]:.2f}", mn,
                        ";".join(st[m] for m in mem)])
    with open(outdir / "hard_review.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "group", "conf_mean", "lowconf_frac"])
        for i, g in sorted(res["hard"], key=lambda x: st[x[0]]):
            w.writerow([st[i], g, f"{conf[i]:.5f}", f"{f['low'][i]:.5f}"])


# ---------------------------------------------------------------- report
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(meta: dict, cfgs: dict, f: dict, names48: dict[int, str], pres48: np.ndarray,
                 px48: np.ndarray, val: dict) -> str:
    N = len(f["stems"])
    partial = N < meta['n_total']
    L = [f"# HackEye curation dry run ({meta['split']})", "",
         (f"**Partial run: {N:,} of {meta['n_total']:,} {meta['split']} frames** have a mask, a labels row and a "
          f"classpix row ({meta['n_masks']:,} masks on disk). Re-run after labelling finishes for the final numbers. "
          if partial else f"**Full run: all {N:,} {meta['split']} frames.** ") +
         f"Nothing in the dataset was changed. Params: max_gap {meta['max_gap']} s, protect_px {meta['protect_px']}, "
         f"rare_frac {meta['rare_frac']}, hard_lowconf {meta['hard_lowconf']}. "
         f"Protected classes: {len(meta['protected'])} ({len(meta['rare_classes'])} rare).", "",
         f"Val is not trimmed: {val['n_masks']:,} masks of {val['n_total']:,} frames.", "",
         "## Sweep", "",
         "| config | stage A | groups | kept | % kept | protected extras | hard-review | mean conf kept | mean conf all |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in cfgs.items():
        L.append(f"| {name} | {c['stage_a']:,} | {c['groups']:,} | {c['kept']:,} | {pct(c['kept'] / N)} | "
                 f"{c['protected_extras']:,} | {c['hard_review']:,} | {c['conf_kept']:.4f} | {meta['conf_all']:.4f} |")
    L += ["", "## Kept per clip", "", "| clip | frames | " + " | ".join(cfgs) + " |",
          "|---|---|" + "---|" * len(cfgs)]
    clips = sorted({c for c in f["clip"]})
    for clip in clips:
        n = sum(1 for c in f["clip"] if c == clip)
        L.append(f"| {clip} | {n:,} | " + " | ".join(f"{cfgs[k]['per_clip'].get(clip, 0):,}" for k in cfgs) + " |")
    L.append("| **total** | " + f"{N:,} | " + " | ".join(f"{cfgs[k]['kept']:,}" for k in cfgs) + " |")
    hl = [k for k in HIGHLIGHT if k in cfgs]
    if hl:
        L += ["", "## Merged classes (48), before vs after", "",
              f"Presence = share of frames with >= {meta['protect_px']} px of the class; pixel share = share of all "
              "pixels. Full tables for every config are in report.json.", "",
              "| class | presence before | " + " | ".join(f"presence {k}" for k in hl) + " | pixels before | "
              + " | ".join(f"pixels {k}" for k in hl) + " |",
              "|---|---|" + "---|" * (2 * len(hl) + 1)]
        for ci, nm in names48.items():
            row = [nm, pct(cfgs[hl[0]]["presence_before"][nm])]
            row += [pct(cfgs[k]["presence_after"][nm]) for k in hl]
            row += [f"{100 * cfgs[hl[0]]['pixshare_before'][nm]:.2f}%"]
            row += [f"{100 * cfgs[k]['pixshare_after'][nm]:.2f}%" for k in hl]
            L.append("| " + " | ".join(row) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/datasets/HackEye"))
    ap.add_argument("--split", default="train", choices=["train", "val"], help="val: counts only, nothing trimmed")
    ap.add_argument("--base-fps", default="2,4")
    ap.add_argument("--sim", default="0.75,0.85,0.90,0.95")
    ap.add_argument("--max-gap", type=float, default=10.0, help="seconds from the group's anchor")
    ap.add_argument("--protect-px", type=int, default=2000)
    ap.add_argument("--rare-frac", type=float, default=0.01)
    ap.add_argument("--hard-lowconf", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vistas-config", default="/datasets/vistas-raw/config_v2.0.json")
    args = ap.parse_args()
    t0 = time.time()
    root = args.root
    out = root / "curation"

    if args.split == "val":
        v = load_frames(root, "val")
        print(f"[curate] val (not trimmed): {len(v['stems'])} usable of {v['n_total']} frames "
              f"({v['n_masks']} masks, {v['n_labels']} labels rows, {v['n_classpix']} classpix rows)")
        return

    tax = load_taxonomy(args.vistas_config)
    merged = build_merge_lut(tax, yaml.safe_load(MERGE_YAML.read_text()))
    lut = merged.lut
    names124 = dict(tax.names)
    names48 = dict(merged.names)
    names48[255] = "(ignore)"

    f = load_frames(root, args.split)
    N = len(f["stems"])
    print(f"[curate] {args.split}: {N} usable of {f['n_total']} frames "
          f"({f['n_masks']} masks, {f['n_labels']} labels rows, {f['n_classpix']} classpix rows)", flush=True)
    if N == 0:
        sys.exit("[curate] nothing to do")
    grids = load_grids(out / f"grid_cache_{args.split}.npz", f["stems"], lut, root / "masks" / args.split, args.workers)

    cm = class_matrix(f["counts"])  # (N,256) Vistas ids
    present = cm >= args.protect_px  # (N,256)
    frac = present[:, :tax.n].mean(0)
    rare = [c for c in range(tax.n) if 0 < frac[c] < args.rare_frac]
    protected = sorted(set(NAV_IDS) | set(rare))
    print(f"[curate] protected classes: {len(protected)} ({len(rare)} rare: {[names124[c] for c in rare]})", flush=True)

    # 48-class pixel counts / presence per frame (before/after tables)
    ids48 = sorted(names48)
    px48 = np.zeros((N, len(ids48)), np.int64)
    col48 = {c: j for j, c in enumerate(ids48)}
    for c in range(tax.n):
        px48[:, col48[int(lut[c])]] += cm[:, c]
    pres48 = px48 >= args.protect_px
    tot_px = px48.sum()

    val = load_frames(root, "val") if (root / "masks" / "val").exists() else {"n_total": 0, "n_masks": 0}
    meta = {"split": args.split, "n_total": f["n_total"], "n_masks": f["n_masks"], "n_used": N, "max_gap": args.max_gap,
            "protect_px": args.protect_px, "rare_frac": args.rare_frac, "hard_lowconf": args.hard_lowconf,
            "protected": [names124[c] for c in protected], "rare_classes": [names124[c] for c in rare],
            "conf_all": float(f["conf"].mean()), "val": {"n_total": val["n_total"], "n_masks": val["n_masks"]}}

    cfgs: dict[str, dict] = {}
    for fps in [int(x) for x in args.base_fps.split(",")]:
        for sim in [float(x) for x in args.sim.split(",")]:
            name = f"fps{fps}_sim{sim:.2f}"
            tc = time.time()
            res = curate(f, grids, present, protected, names124, fps, sim, args.max_gap, args.hard_lowconf)
            write_config(out / name, args.split, f, res)
            ki = np.array([i for i, _, _ in res["keep"]])
            per_clip = defaultdict(int)
            for i in ki:
                per_clip[f["clip"][i]] += 1
            cfgs[name] = {
                "stage_a": res["n_stage_a"], "groups": len(res["groups"]), "kept": len(ki),
                "protected_extras": sum(1 for _, _, w in res["keep"] if w != "best"),
                "hard_review": len(res["hard"]), "conf_kept": float(f["conf"][ki].mean()),
                "per_clip": dict(per_clip),
                "presence_before": {names48[c]: float(pres48[:, j].mean()) for c, j in col48.items()},
                "presence_after": {names48[c]: float(pres48[ki, j].mean()) for c, j in col48.items()},
                "pixshare_before": {names48[c]: float(px48[:, j].sum() / tot_px) for c, j in col48.items()},
                "pixshare_after": {names48[c]: float(px48[ki, j].sum() / px48[ki].sum()) for c, j in col48.items()},
            }
            print(f"[curate] {name}: stage A {res['n_stage_a']}, groups {len(res['groups'])}, kept {len(ki)} "
                  f"({100 * len(ki) / N:.1f}%), extras {cfgs[name]['protected_extras']}, "
                  f"hard {len(res['hard'])}  [{time.time() - tc:.1f}s]", flush=True)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps({"meta": meta, "configs": cfgs}, indent=1))
    (out / "report.md").write_text(build_report(meta, cfgs, f, {c: names48[c] for c in ids48}, pres48, px48, val))
    print(f"[curate] done in {time.time() - t0:.1f}s -> {out}/report.md", flush=True)


if __name__ == "__main__":
    main()
