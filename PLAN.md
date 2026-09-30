> **Note:** this plan was authored and the code below was written while this
> project lived at `HackEye/training/` inside the main `HackEye` repo — see
> that plan's design constraint (self-contained, no imports from `HackEye`,
> extractable via a clean move). It has since been extracted to this dedicated
> repo, exactly as planned. References to `training/` below refer to what is
> now this repo's root. **See `PROGRESS.md` for current status and the exact
> next step** — this file is the original design record and is mostly
> historical from here on; update it only if the approach itself changes.

# Train yolo26s-sem on Mapillary Vistas v2.0 (124 classes)

## Context

HackEye detects pedestrian-safety boundaries from street footage. The current
segmentation stage uses the **19-class Cityscapes** taxonomy
(`common/config/safety_mapping.json` in the main `HackEye` repo), which
separates `road` from `sidewalk` but has **no curb class at all** — a
repo-wide grep for "curb" returns zero hits.

Curbs are the actual object of interest: the curb *is* the boundary between road
and sidewalk, and boundary precision is what this project cares about. Mapillary
Vistas v2.0 labels `curb` and `curb cut` natively, alongside `pedestrian area`,
`bike lane`, `pothole` and split crosswalk markings.

This plan covers **conversion + training only**: build a Vistas→Ultralytics
converter, train `yolo26s-sem` on the full 124-class taxonomy, and validate it.
Rewiring the downstream safety pipeline (`safety_mapping.json`,
`RnD/src/segmentation.py`, Swift `SafetySegmentation` — all in the `HackEye`
repo) is a deliberate follow-up, tracked separately from this repo.

### Fixed decisions

| Decision | Choice |
|---|---|
| Taxonomy | Vistas **v2.0, full 124 classes** |
| Dataset | Downloaded, extracted, converted — see PROGRESS.md |
| Scope | Conversion + training only; no `HackEye` repo changes |
| Init weights | Warm-start from a pretrained `-sem` checkpoint (see below) |
| Code location | This repo (originally `HackEye/training/`, extracted per its own design) |

## Verified findings that shape the implementation

All of the following were read from `ultralytics 8.4.167` **inside the actual
built image**, not inferred:

1. **The `masks_dir` trap.** `data/build.py:271-275` selects
   `SemanticDataset` if the YAML has a `masks_dir:` key, else
   `PolygonSemanticDataset` — which injects an extra `background` class and
   silently makes `nc` 125. **Always set `masks_dir`.**
2. **Image paths must contain a literal `/images/` component.**
   `img2label_paths` (`data/utils.py:131-143`) derives mask paths by string-replacing
   the *last* `/images/` with `/<masks_dir>/` and forcing a `.png` extension. The
   mask tree is never globbed; it is entirely implied by the image tree.
3. **Mask format:** single-channel 8-bit PNG, mode `P` (palette index used as
   class id, RGB ignored) or `L`. **Ignore index is hardcoded 255** across the
   loss, metrics, and every augmentation border fill.
4. **Hard assertion `mask.shape[:2] == image.shape[:2]`** (`data/utils.py:399`).
   Any resize must be applied identically to both.
5. **Vistas needs no pixel remapping.** Vistas v2.0 ships `v2.0/labels/*.png` as
   8-bit palette PNGs whose values are already indices 0..123 into
   `config_v2.0.json`, with 123 = `Unlabeled`. The converter is a
   **file-layout + resize job, not a decode job.**
6. **YAML: supply `names` only, never `nc`.** If both are present they must match
   exactly or it raises; `nc` is derived from `len(names)`.
7. **Training `imgsz` must be a scalar.** `engine/trainer.py:351` calls
   `check_imgsz(..., max_dim=1)`, which warns and silently collapses a tuple to
   `max(imgsz)`. Only `predict`/`export` accept `(h, w)`. Variable aspect is
   handled by `SemanticDataset.load_image` (short side → `imgsz` when
   training), so `rect` is unnecessary.
8. **No head surgery needed.** `SemanticSegmentationTrainer.get_model` builds at
   the dataset's `nc` *then* loads weights through `intersect_dicts`, which drops
   shape-mismatched keys. Exactly **4 tensors** (`classifier.1.{weight,bias}`,
   `aux_head.1.{weight,bias}`) stay at fresh init; everything else transfers.
   **Confirmed empirically** in the real smoke train: log line read
   `Transferred 360/364 items` — deficit exactly 4.
9. **Name-based head remapping does not apply.** `_remap_cls_by_names` keys off
   `isinstance(m, Detect)`; `SemanticSegment` subclasses `nn.Module`. Don't try to
   match Vistas names to ADE20K names hoping for row transfer.
10. **124 classes costs almost no VRAM.** The head emits logits at stride 8, so
    at 640/batch 16 that tensor is ~51 MB. Memory is dominated by
    `nc`-independent backbone activations.
11. **`SemanticMetrics` API** (used by `eval_semantic.py`, confirmed by reading
    `ultralytics/utils/metrics.py` directly): `.miou`, `.pixel_accuracy`,
    `.per_class_iou` (np.ndarray by class id), `.per_class_pixel_accuracy`,
    `.nt_per_class`, `.nt_per_image`, `.ap_class_index` (classes present in
    GT), `.matrix` (confusion, `[gt, pred]` layout), `.summary()` /
    `.to_csv()` (via `polars`, already an ultralytics transitive dep — no new
    dependency needed). `model.val()` returns this object directly.
12. **`cls_pw` defaults to `0.0`, which fully disables class weighting** —
    `DetectionTrainer.set_class_weights` returns immediately if
    `cls_pw == 0.0`, without even scanning masks. At `cls_pw=1.0`, expect a
    ~2-4 min startup cost (full single-pass mask scan for `get_class_counts`).
    **Confirmed empirically**: the smoke train printed
    `Class weights: [0.064  0.853  1.377  1.539  0.088  1.539  1.539]`.
13. **Early-abort hook**: `model.add_callback("on_pretrain_routine_end", fn)`
    fires right after model build + weight load + dataloader setup, and
    *before* the epoch loop starts (`_setup_train` → callback → `_do_train`).
    This is what makes a strict transfer-count check useful on the full
    100-epoch run and not just the smoke test.

## Prerequisite (human step)

Vistas cannot be fetched by script — register and accept the research license
at <https://www.mapillary.com/dataset/vistas>. **Done** — see PROGRESS.md for
exact paths and verified counts.

## Implementation

### Layout

```
(this repo's root)
  vistas_config.py      # parse config_v2.0.json -> names, palette, evaluate flags, void ids
  vistas_convert.py     # converter + dataset YAML emitter
  vistas_verify.py      # pre-training correctness checks (5 subcommands)
  train_semantic.py     # training entrypoint
  eval_semantic.py      # per-class IoU report
  gpu_guard.py           # CUDA/VRAM fail-fast checks
  requirements.txt
  Dockerfile
  compose.yaml
  weights/               # gitignored — cached checkpoints
  PLAN.md                # this file
  PROGRESS.md            # current status, read this first
```

`Dockerfile` uses `FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` directly
— chosen because Blackwell (RTX 50-series, sm_120) needs CUDA 12.8+.

### Derived dataset layout

```
/home/megabeast/datasets/vistas-yolo/
  vistas-v2.0.yaml
  images/{train,val}/<stem>.jpg
  masks/{train,val}/<stem>.png
  conversion_manifest.json
```

The `images/` component is mandatory (finding #2). The YAML uses an
**absolute `path:`** so it doesn't resolve against Ultralytics'
`datasets_dir` setting.

### Converter behavior

- **No pixel remapping** by default (finding #5). Keeps Vistas' native 0..123
  indices so every debug print and overlay stays comparable to the official
  taxonomy. `--drop-void` renumbers away void classes instead, if ever needed
  (palette is renumbered in lockstep — this was a real bug, fixed during
  testing, see PROGRESS.md).
- `nc = 124`; `names` 1:1 from `config_v2.0.json`, with a `label_mapping`
  sending every class with `"evaluate": false` to `ignore_label` — **derived
  from the config at runtime, never hardcoded** (turned out to matter: the
  real config has 8 void classes, not the 1-2 originally guessed).
- Class names: last **two** `--`-hierarchy components joined
  (`construction--barrier--curb` → `barrier curb`); falls back to the full
  name on collision.
- **Resize offline**: image `INTER_AREA`, mask `INTER_NEAREST`, shape lockstep
  asserted before writing. Default `--short-side 768`, `--max-long-side 2048`.
- `--short-side 0` = passthrough via symlink, for round-trip verification.
- `--limit N` builds a smoke subset through the identical code path.
- Parallelized with stdlib `concurrent.futures.ProcessPoolExecutor`.

### Warm-start checkpoint

Both exist at tag `v8.4.0`: `yolo26s-sem.pt` (Cityscapes-19, 13.25MB) and
`yolo26s-sem-ade20k.pt` (ADE20K-150, 13.29MB, **the default** — ADE20K's
150-category pretraining is judged more likely to retain fine texture/part
features relevant to `curb` vs `sidewalk` than Cityscapes-19, which has no
curb concept at all. This is judgment, not verification — **A/B it** before
committing to the full run, see the A/B step below).

`train_semantic.py` captures the "Transferred X/Y items" log line via a
logging handler and asserts the deficit is exactly 4 (finding #8) through an
early-abort callback (finding #13) — catches a wrong checkpoint/architecture
before wasting any epoch time, on both the smoke run and the full run.

### Training config (defaults in `train_semantic.py`)

```python
YOLO("yolo26s-sem-ade20k.pt").train(
    data="/datasets/vistas-yolo/vistas-v2.0.yaml",
    task="semantic", epochs=100,
    imgsz=640,            # scalar only (finding #7)
    batch=16, amp=True, workers=8,
    cls_pw=1.0,           # see finding #12 — critical, NOT ultralytics' own default
    close_mosaic=10, cos_lr=True, optimizer="auto", device=0,
    project="/runs", name="vistas124-s-640",
    save_period=10, plots=True, seed=0, deterministic=False,
)
```

**Why `cls_pw=1.0`:** at the Ultralytics default of `0.0`, class weighting is
skipped entirely. With `road` at ~18% of pixels and `curb cut` at ~0.1%,
unweighted CE would predict `curb cut` essentially never — and those classes
are the entire reason for this project. The ENet inverse-log formula is
bounded (~35× max ratio), so this isn't a stability risk.

**Compose sets two things a naive setup misses:**
- `shm_size: "8gb"` — Docker's 64MB default kills DataLoader workers with
  `Bus error` once you're actually moving batches through shared memory.
- `YOLO_CONFIG_DIR` pointed at a writable bind mount — with a non-root
  container user, Ultralytics otherwise falls back to `/tmp` for its
  settings/weights cache, which evaporates on container exit. **Observed
  directly** when first probing the image.

### VRAM (estimates — validate on the real run)

| imgsz | batch | est. peak | verdict |
|---|---|---|---|
| 640 | 16 | ~7–9 GB | **recommended start** |
| 768 | 16 | ~10–13 GB | probably fits |
| 1024 | 8 | ~11–14 GB | fits |
| 1024 | 16 | ~20+ GB | will not fit |

Card is a single RTX 5060 Ti, 16GB. Drop batch before dropping imgsz if more
resolution is needed; keep `batch >= 8`.

### Wall-clock

**~3–5 min/epoch → ~6–9 hours for 100 epochs** (an overnight run) at
`imgsz=640`, pre-resized to short-side 768. Offline resizing is roughly a 5×
speedup over training at native resolution (which would be 25–50 hours) — the
single highest-leverage decision in this plan.

## Verification

1. **Converter correctness** (`vistas_verify.py roundtrip`, `shapes`,
   `yaml-check`) — round-trip identity in passthrough mode, subset-check in
   resize mode, shape lockstep + range check across the whole dataset, and
   confirmation that the YAML selects `SemanticDataset` not
   `PolygonSemanticDataset`.
2. **Visual overlay** (`vistas_verify.py overlay`) — blend images with
   palette-colorized masks; curb should trace a thin ribbon exactly along the
   sidewalk/road boundary.
3. **Resolution retention study** (`vistas_verify.py retention`) — see
   PROGRESS.md for the actual result on real data (it defied the original
   assumption: curb/curb-cut *gain* relative share under downsampling, no
   cliff found even down to short-side 192).
4. **Smoke train** — `vistas8` (`--limit 8`) through the full 124-class
   pipeline, 20 epochs, `batch=4`. Pass criteria: `0 missing masks, 0
   corrupt`; exactly 4 tensors missing from transfer; no NaN; val mIoU well
   above 0 (should overfit hard on 8 images). **Gate — do not skip before the
   overnight run.**
5. **A/B** (~1-3h): 10 epochs each of `{cityscapes-init, ade20k-init} ×
   {cls_pw=0, cls_pw=1}`. **Priority reordered 2026-09-30** (see
   PROGRESS.md): this round judges candidates on per-class IoU for the
   broad navigable-surface classes — `road`/`sidewalk`/`bike-lane` and
   similar — not curb/curb-cut. Curb/curb-cut fine-tuning is a deliberate
   later pass; `cls_pw=1` exists specifically to boost those rare classes
   and may cost accuracy on the common ones, so expect `cls_pw=0` to win
   this round unless the data says otherwise.
6. **Final evaluation** (`eval_semantic.py`) — headline mIoU + pixel accuracy,
   a dedicated table for the four target classes (the real acceptance
   criterion, not mIoU), and the 4×4 confusion sub-block. Calibration:
   published Vistas v2.0 mIoU for heavyweight models on the full taxonomy is
   low-to-mid 40s; a ~10M-param model at 640 landing high-20s to mid-30s is
   reasonable — judge on the target classes, not the headline number.

## Explicitly out of scope

No changes to `HackEye`'s `common/config/safety_mapping.json`,
`RnD/src/segmentation.py`, or any Swift code — that rewiring is a deliberate
follow-up once a trained checkpoint exists.

**Useful seam for that follow-up:** `results[0].semantic_mask.data` stays a
per-pixel argmax grid regardless of `nc`, so **a 124-class checkpoint
satisfies `HackEye`'s existing inference contract unchanged** — only the
*meaning* of the integers changes. `RnD/src/segmentation.py` currently
hard-rejects any model whose class count isn't 19, and `models/manifest.json`
still describes the segmentation checkpoint as LRASPP-MobileNetV3 while
`RnD/src/models.py` now loads a YOLO semantic model — both worth reconciling
when the rewiring happens.
