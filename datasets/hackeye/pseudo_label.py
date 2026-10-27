"""Pseudo-labels the extracted HackEye frames with the dataset-creation model
(Yolo26x-sem-Vistas124-HackEye-1280) and finalizes the dataset: masks, per-frame
confidence stats, frames.csv, HackEye.yaml and dataset_manifest.json.

    <dst>/masks/{train,val}/<stem>.png   8-bit palette PNG, class ids 0..123, Vistas colours
    <dst>/meta/labels_<split>.csv        stem,conf_mean,conf_p10,lowconf_frac
    <dst>/meta/classpix_<split>.csv      stem,counts   (sparse "id:pixels;id:pixels" per frame)
    <dst>/meta/class_pixels.json         running per-class pixel totals
    <dst>/frames.csv, HackEye.yaml, dataset_manifest.json   (finalize)

Why it is built the way it is:

* The masks are PSEUDO-labels, to be curated by hand later. Per-frame teacher
  confidence (mean max-softmax, 10th percentile, share of pixels < 0.5) is
  stored so curation can sort frames by how unsure the teacher was. Full
  confidence maps are not stored (~30 GB).
* A custom SemanticSegmentationPredictor subclass is used because the stock
  postprocess throws the logits away after argmax; here the softmax is taken
  on the GPU while the logits are still there. This is done one image at a
  time inside postprocess: a [124, 720, 1280] float32 tensor is ~460 MB, so
  batching the softmax would not fit next to the model on a 16 GB card.
  Upsample + scale_masks are done exactly like the parent class.
* Multi-scale: the teacher's softmax probabilities are averaged over the
  short sides in --imgsz (default 736,1280). A scale study on 500 Vistas val
  images (downscaled to 736 short side and back, see PLAN.md) gave mIoU 0.422
  for the mean-softmax ensemble vs 0.402 @736 and 0.424 @1280, with the best
  pixel accuracy and the most balanced navigable surfaces, at less than the
  cost of 1280 alone being "free". Each scale is a separate predict() call on
  the same chunk; the predictor keeps one running (fp16) mean-probability
  buffer per frame, and only the last scale pass takes max/stats/class map.
* Batch size: measured at 736,1280 on 40 frames (1280x720): batch 1 / 4 / 8 =
  4.1 / 3.8 / 3.7 frames/s, peak VRAM 2.2 / 4.9 / 9.5 GiB. The per-image
  softmax work dominates, so batching buys nothing; default is 1.
* The network input size per scale follows diagnose_surfaces.py
  --match-train-scale: the short side the net sees is that scale (the X model
  trained at short side 1280), each dimension rounded up to a multiple of 32.
* Resumable: PNGs are written atomically (tmp + os.replace) in a thread pool so
  the GPU is not starved by PNG encoding. A frame counts as done only when its
  mask exists AND its stats rows are in the csvs; the stats rows are appended
  only after the PNG is on disk, so a crash can only produce "mask without
  stats" (redone) or never "stats without mask".
* class_pixels resume-safety: pixel counts are stored per frame in
  meta/classpix_<split>.csv (sparse, ~1 KB/frame), appended in the same step
  as the confidence stats. Totals are never accumulated in memory across
  runs: on start they are rebuilt from those csvs (last row per stem wins,
  truncated trailing lines from a kill are ignored), then updated as frames
  complete, and class_pixels.json is rewritten every few thousand frames and
  at the end. So any restart yields correct totals, with no double counting.
* Finalize fails loudly if any jpg lacks a mask (or the other way round),
  or if frame/csv rows disagree. A dataset with silent holes would corrupt
  the curation and every training run on it.

Run (GPU, in the train container), e.g.

    cd docker && docker compose run --rm --entrypoint python train \\
        datasets/hackeye/pseudo_label.py --imgsz 736,1280
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vistas" / "conversion"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

from gpu_guard import assert_torch_cuda  # noqa: E402
from vistas_config import load_taxonomy  # noqa: E402

SPLITS = ("train", "val")
LABEL_COLS = ["stem", "conf_mean", "conf_p10", "lowconf_frac"]
LOWCONF = 0.5
SAVE_EVERY = 3000  # frames between class_pixels.json rewrites


# ----------------------------------------------------------------- helpers

def _palette_bytes(palette: dict[int, tuple[int, int, int]], n: int) -> list[int]:
    flat: list[int] = []
    for cid in range(n):
        flat.extend(palette[cid])
    flat.extend([0, 0, 0] * (256 - n))
    return flat[:768]


def _write_mask_png(path: Path, arr: np.ndarray, palette: list[int]) -> None:
    h, w = arr.shape
    im = Image.frombytes("P", (w, h), np.ascontiguousarray(arr).tobytes())
    im.putpalette(palette)
    tmp = path.with_name(path.name + ".tmp")
    im.save(tmp, format="PNG")
    os.replace(tmp, path)


def _net_size(h: int, w: int, imgsz: int) -> tuple[int, int]:
    k = imgsz / min(h, w)
    return (int(math.ceil(h * k / 32) * 32), int(math.ceil(w * k / 32) * 32))


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_csv_rows(path: Path) -> list[dict]:
    """Reads a csv, ignoring a truncated final line left by a killed run."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in body if len(r) == len(header)]


def _open_append(path: Path, header: list[str]):
    """Opens a csv for appending; writes the header if new, and repairs a missing
    trailing newline (killed mid-line) so the next row starts on its own line."""
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "ab") as g:
                    g.write(b"\n")
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w


def _encode_counts(counts: np.ndarray) -> str:
    nz = np.flatnonzero(counts)
    return ";".join(f"{i}:{int(counts[i])}" for i in nz)


def _decode_counts(s: str, nc: int, into: np.ndarray) -> None:
    if not s:
        return
    for part in s.split(";"):
        i, c = part.split(":")
        into[int(i)] += int(c)


def _load_totals(meta: Path, nc: int) -> tuple[np.ndarray, dict[str, set[str]]]:
    """Rebuilds class totals + the set of stems with complete stats from disk."""
    totals = np.zeros(nc, np.int64)
    done: dict[str, set[str]] = {}
    for split in SPLITS:
        lab = {r["stem"] for r in _read_csv_rows(meta / f"labels_{split}.csv")}
        cp: dict[str, str] = {r["stem"]: r["counts"] for r in _read_csv_rows(meta / f"classpix_{split}.csv")}
        stems = lab & cp.keys()
        for s in stems:
            _decode_counts(cp[s], nc, totals)
        done[split] = stems
    return totals, done


def _save_totals(meta: Path, totals: np.ndarray, names: dict[int, str]) -> None:
    _atomic_text(meta / "class_pixels.json", json.dumps(
        {"total_pixels": int(totals.sum()),
         "classes": {str(i): {"name": names[i], "pixels": int(totals[i])} for i in range(len(totals))}},
        indent=1))


# -------------------------------------------------------------- prediction

def make_predictor_class():
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.semantic.predict import SemanticSegmentationPredictor
    from ultralytics.utils import ops

    class StatsPredictor(SemanticSegmentationPredictor):
        """Same class map as the parent, plus per-frame confidence stats and
        class pixel counts attached to each Results as `.hk` (dict).

        Multi-scale: state lives on the class (the Model owns the instance, so
        the main loop cannot easily set instance attributes). Every pass adds
        probs/n_scales into acc[path]; only the pass with `final=True` builds
        the class map and stats, then drops the buffer."""

        acc: dict = {}
        final = True
        n_scales = 1
        net_hw_seen: set = set()

        def postprocess(self, preds, img, orig_imgs):
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            results = []
            for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
                img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
                pred = pred[None].float()
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="bilinear", align_corners=False)
                pred = ops.scale_masks(pred, orig_img.shape[:2])[0]   # [nc, H, W] logits
                nc = pred.shape[0]
                probs = pred.softmax(0)
                del pred
                self.net_hw_seen.add(tuple(img.shape[2:]))
                key = str(img_path)
                part = probs.div_(self.n_scales).half()
                del probs
                if key in self.acc:
                    self.acc[key].add_(part)
                    del part
                else:
                    self.acc[key] = part
                if not self.final:  # intermediate scale: nothing to report yet
                    results.append(Results(orig_img, path=img_path, names=self.model.names,
                                           semantic_mask=torch.zeros(1, 1, dtype=torch.uint8)))
                    continue
                probs = self.acc.pop(key).float()
                conf, cls = probs.max(0)
                del probs
                class_map = cls.to(torch.uint8)
                flat = conf.flatten()
                hk = {
                    "conf_mean": float(flat.mean()),
                    "conf_p10": float(torch.quantile(flat, 0.1)),
                    "lowconf_frac": float((flat < LOWCONF).float().mean()),
                    "counts": torch.bincount(cls.flatten(), minlength=nc).cpu().numpy(),
                    "net_hw": tuple(img.shape[2:]),
                }
                del conf, cls, flat
                r = Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map)
                r.hk = hk
                results.append(r)
            return results

    return StatsPredictor


# ---------------------------------------------------------------- finalize

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


def finalize(args, tax, names: dict[int, str]) -> None:
    scales = [int(x) for x in str(args.imgsz).split(",")]
    dst: Path = args.dst
    meta = dst / "meta"
    nc = tax.n

    # 1. every jpg has a mask and vice versa
    problems = []
    for split in SPLITS:
        jpg = {p.stem for p in (dst / "images" / split).glob("*.jpg")}
        png = {p.stem for p in (dst / "masks" / split).glob("*.png")}
        if jpg - png:
            problems.append(f"{split}: {len(jpg - png)} jpg without mask, e.g. {sorted(jpg - png)[:3]}")
        if png - jpg:
            problems.append(f"{split}: {len(png - jpg)} mask without jpg, e.g. {sorted(png - jpg)[:3]}")
    if problems:
        sys.exit("[finalize] FAILED: " + "; ".join(problems))

    # 2. merge frame csvs + label csvs
    frame_rows: list[dict] = []
    for p in sorted(meta.glob("frames_*.csv")):
        frame_rows += _read_csv_rows(p)
    labels: dict[str, dict] = {}
    for split in SPLITS:
        for r in _read_csv_rows(meta / f"labels_{split}.csv"):
            labels[r["stem"]] = r  # last row wins (re-labelled frames)
    stems = [r["stem"] for r in frame_rows]
    if len(set(stems)) != len(stems):
        sys.exit("[finalize] FAILED: duplicate stems across frames_*.csv")
    n_img = sum(1 for s in SPLITS for _ in (dst / "images" / s).glob("*.jpg"))
    if n_img != len(frame_rows):
        sys.exit(f"[finalize] FAILED: {n_img} jpgs on disk vs {len(frame_rows)} rows in frames_*.csv")
    miss = [s for s in stems if s not in labels]
    if miss:
        sys.exit(f"[finalize] FAILED: {len(miss)} frames without label stats, e.g. {miss[:3]}")

    cols = list(frame_rows[0].keys()) + LABEL_COLS[1:]
    frame_rows.sort(key=lambda r: (r["split"], r["stem"]))
    with open(dst / "frames.csv.tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in frame_rows:
            lab = labels[r["stem"]]
            w.writerow([r[c] for c in cols[:-3]] + [lab[c] for c in LABEL_COLS[1:]])
    os.replace(dst / "frames.csv.tmp", dst / "frames.csv")

    # 3. yaml
    data = tax.to_data_yaml_dict(root=str(dst))
    data["path"] = "/datasets/HackEye"
    data["train"], data["val"] = "images/train", "images/val"
    _atomic_text(dst / "HackEye.yaml", yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    # 4. manifest
    totals, _ = _load_totals(meta, nc)
    _save_totals(meta, totals, names)
    clips: dict[str, dict] = {}
    for r in frame_rows:
        c = clips.setdefault(r["clip"], {"clip": r["clip"], "split": r["split"], "src_res": f"{r['src_w']}x{r['src_h']}",
                                         "src_fps": float(r["src_fps"]), "frames": 0, "last_t": 0.0})
        c["frames"] += 1
        c["last_t"] = max(c["last_t"], float(r["t_sec"]))
    for c in clips.values():
        c["duration_s"] = round(c.pop("last_t"), 3)
    first = frame_rows[0]
    fps_out = 20.0
    if len(frame_rows) > 1 and first["clip"] == frame_rows[1]["clip"]:
        fps_out = round(1.0 / (float(frame_rows[1]["t_sec"]) - float(first["t_sec"])), 6)
    sample = next(iter((dst / "images" / "train").glob("*.jpg")), None) or next(iter((dst / "images" / "val").glob("*.jpg")))
    with Image.open(sample) as im:
        img_w, img_h = im.size
    wp = Path(args.weights)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(wp), "weights_sha256": _sha256(wp),
        "imgsz_short_sides": scales, "net_input_hw": [list(_net_size(img_h, img_w, z)) for z in scales],
        "ensemble": "mean softmax over scales",
        "git_commit": _git_commit(Path("/app")),
        "extraction": {"fps": fps_out, "width": img_w, "height": img_h, "resize": "cv2.INTER_AREA",
                       "frame_pick": "src_frame = round(k / fps * src_fps)",
                       "jpeg_quality": args.jpeg_quality, "jpeg_optimize": True},
        "clips": sorted(clips.values(), key=lambda c: c["clip"]),
        "counts": {s: sum(1 for r in frame_rows if r["split"] == s) for s in SPLITS} | {"total": len(frame_rows)},
        "class_pixels": {names[i]: int(totals[i]) for i in range(nc)},
        "class_pixels_total": int(totals.sum()),
    }
    _atomic_text(dst / "dataset_manifest.json", json.dumps(manifest, indent=1))
    print(f"[finalize] ok: {len(frame_rows)} frames, frames.csv + HackEye.yaml + dataset_manifest.json in {dst}", flush=True)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", type=Path, default=Path("/datasets/HackEye"))
    ap.add_argument("--weights", default="models/Yolo26x-sem-Vistas124-HackEye-1280.pt")
    ap.add_argument("--imgsz", default="736,1280", help="comma list of short sides the net sees; softmax is averaged over them (each dim rounded up to /32)")
    ap.add_argument("--batch", type=int, default=1, help="frames per predict call; 1 was fastest and lightest at 736,1280 (see docstring)")
    ap.add_argument("--vistas-config", default="/datasets/vistas-raw/config_v2.0.json")
    ap.add_argument("--limit", type=int, default=0, help="label at most N frames this run (smoke tests)")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--png-threads", type=int, default=4)
    ap.add_argument("--jpeg-quality", type=int, default=100, help="recorded in the manifest only (extract_frames.py sets it)")
    args = ap.parse_args()

    tax = load_taxonomy(args.vistas_config)
    names, nc = dict(tax.names), tax.n
    dst, meta = args.dst, args.dst / "meta"
    palette = _palette_bytes(tax.palette, nc)

    if args.finalize_only:
        finalize(args, tax, names)
        return

    device = assert_torch_cuda()
    from ultralytics import YOLO

    images = []
    for split in SPLITS:
        d = dst / "images" / split
        images += [(split, p) for p in sorted(d.glob("*.jpg"))]
        (dst / "masks" / split).mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    totals, done = _load_totals(meta, nc)
    todo = [(s, p) for s, p in images
            if not (p.stem in done[s] and (dst / "masks" / s / f"{p.stem}.png").exists())]
    total_all = len(images)
    already = total_all - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[label] {total_all} frames, {already} already done, {len(todo)} to label this run", flush=True)

    if todo:
        with Image.open(todo[0][1]) as im:
            W, H = im.size
        scales = [int(x) for x in args.imgsz.split(",")]
        szs = [_net_size(H, W, z) for z in scales]
        print(f"[label] frame {W}x{H}, ensemble (mean softmax) over short sides {scales}, net inputs "
              f"{[f'{z[1]}x{z[0]}' for z in szs]}, batch {args.batch}", flush=True)

        model = YOLO(args.weights)
        Predictor = make_predictor_class()

        lab_f, lab_w, cp_f, cp_w = {}, {}, {}, {}
        for s in SPLITS:
            lab_f[s], lab_w[s] = _open_append(meta / f"labels_{s}.csv", LABEL_COLS)
            cp_f[s], cp_w[s] = _open_append(meta / f"classpix_{s}.csv", ["stem", "counts"])

        pool = ThreadPoolExecutor(max_workers=args.png_threads)
        pending: deque = deque()  # (future, split, stem, hk)
        n_done = 0
        t0 = time.time()
        last_save = 0

        def drain(block_until: int) -> None:
            nonlocal n_done, last_save
            while pending and (len(pending) > block_until or pending[0][0].done()):
                fut, split, stem, hk = pending.popleft()
                fut.result()  # raises on PNG failure
                lab_w[split].writerow([stem, f"{hk['conf_mean']:.5f}", f"{hk['conf_p10']:.5f}", f"{hk['lowconf_frac']:.5f}"])
                cp_w[split].writerow([stem, _encode_counts(hk["counts"])])
                lab_f[split].flush()
                cp_f[split].flush()
                totals[:] += hk["counts"]
                n_done += 1
                if n_done % 500 == 0:
                    el = time.time() - t0
                    rate = n_done / el
                    print(f"[label] {already + n_done}/{total_all} done, {rate:.2f} frames/s, "
                          f"ETA {(len(todo) - n_done) / rate / 60:.0f} min", flush=True)
                if n_done - last_save >= SAVE_EVERY:
                    _save_totals(meta, totals, names)
                    last_save = n_done

        try:
            for b in range(0, len(todo), args.batch):
                chunk = todo[b: b + args.batch]
                Predictor.acc.clear()
                Predictor.n_scales = len(szs)
                for si, sz in enumerate(szs):
                    Predictor.final = si == len(szs) - 1
                    gen = model.predict([str(p) for _, p in chunk], imgsz=sz, batch=args.batch, verbose=False,
                                        device=device.index if device.index is not None else 0,
                                        predictor=Predictor, stream=True)
                    # Exhaust the generator: ultralytics holds a non-reentrant lock for as long as
                    # a stream_inference generator is suspended, so a half-consumed one deadlocks
                    # the next predict() call.
                    final_results = list(gen)
                for (split, p), r in zip(chunk, final_results):
                    cm = r.semantic_mask.data
                    assert tuple(cm.shape) == (H, W), (p.name, tuple(cm.shape), (H, W))
                    arr = cm.cpu().numpy()
                    hk = r.hk
                    fut = pool.submit(_write_mask_png, dst / "masks" / split / f"{p.stem}.png", arr, palette)
                    pending.append((fut, split, p.stem, hk))
                    del r, cm
                    drain(block_until=64)
            drain(block_until=0)
            seen = sorted(Predictor.net_hw_seen)
            print(f"[label] net input sizes actually seen (h, w): {seen}", flush=True)
            if len(seen) != len(set(szs)):
                sys.exit(f"[label] FAILED: expected net inputs {sorted(set(szs))}, saw {seen}")
        finally:
            pool.shutdown(wait=True)
            for s in SPLITS:
                lab_f[s].close()
                cp_f[s].close()
            _save_totals(meta, totals, names)
        el = time.time() - t0
        print(f"[label] finished {n_done} frames in {el:.0f}s ({n_done / max(el, 1e-9):.2f} frames/s), "
              f"peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    if args.limit:
        print("[label] --limit given: skipping finalize (run again with --finalize-only)", flush=True)
        return
    finalize(args, tax, names)


if __name__ == "__main__":
    main()
