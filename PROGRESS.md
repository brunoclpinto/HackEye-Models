# Current status

**Read this first.** See `PLAN.md` for the full design/rationale. This file
tracks what's actually been done and the exact next step, so a fresh session
can resume without re-deriving context.

Last updated: 2026-10-26.

**Models in `models/` (naming changed 2026-10-26):**

| File | Role | Training |
|---|---|---|
| `Yolo26m-sem-HackEye26.1-48-1280.pt` | **App release** (HackEye v26.1), the model the app runs | m, 48-class merged taxonomy, 1280; the former `Yolo26m-sem-HackEye20261011-1280.pt` |
| `Yolo26x-sem-Vistas124-HackEye-1280.pt` | **Dataset creation**: base or teacher model for curating and pre-labelling training images from the HackEye media library. Not an app release (too heavy for A14). | x, full 124-class Vistas, 1280; see "X-1280 run" below |

Naming: app releases are `Yolo26<size>-sem-HackEye<app version>-<classes>-<train res>.pt`.
Dataset-creation models are `Yolo26<size>-sem-<dataset><classes>-HackEye-<train res>.pt`, with no
app version. The 48-class merge map shares the app release's name:
`datasets/vistas/conversion/merges/HackEye26.1-48.yaml` (formerly `merges/v1.yaml`). (The scheme is inferred from the two names above; correct it here if the
intent differs.)

The 544 fine-tune (`Yolo26m-sem-HackEye20261012-544.pt`) and the 124-class
s@640 `Yolo26s-sem-HackEyeV1.pt` were removed on 2026-10-18: the user judged
the 544 model too flimsy in the Lab, and V1 was superseded. Both are still in
git history. Next: curate the pseudo-labelled HackEye frames dataset (see "HackEye frames dataset" below and
`datasets/hackeye/README.md`), then
retrain the app model on Vistas plus those frames (720p/l is the leading candidate, ~140 ms on A14).

**Run outputs torn down again 2026-10-26** after the X-1280 release: the whole `vistas124-x-1280` run dir
(about 20 GB of epoch checkpoints and the epoch-81 backup), the X probes (`x-probe`), the scale checks
(`scale-check`, `scale720.*`), the X-1280 launch log and email helper, and the derived datasets
`vistas-yolo-1280` (124-class 1280 train + val) and `vistas-yolo-1280-merge-v1`. Only `vistas-raw` and
`runs/ultralytics-cfg` remain; rebuild with `training/launch_x1280.sh` or the commands below. The repo's untracked
`weights/yolo26n.pt` (Ultralytics' AMP-check download) and an empty `runs/semantic/val` were removed too.

**Storage (since 2026-10-18):** datasets and run outputs live outside the
repo, in `~/media/hackeye-models/{datasets,runs}` (mounted at `/datasets` and
`/runs` by `docker/compose.yaml`; override with `HACKEYE_DATASETS_DIR` /
`HACKEYE_RUNS_DIR`). The repo holds only definition files. Older paths below
(`training/datasets`, `training/runs`, `/home/megabeast/datasets`) are
historical. X-1280 launch script: `training/launch_x1280.sh`.

**Run outputs torn down 2026-10-11, and again 2026-10-12** after the 544
fine-tune (its run folder, checkpoints, eval outputs, `val540-orientation`
and the rebuilt datasets all deleted; numbers are recorded below, release
is in `models/`). Everything under `training/runs/`
(`/runs/` in the container) was deleted except `ultralytics-cfg/` (required
by `YOLO_CONFIG_DIR`). Paths to `/runs/...` below are historical: the
numbers are recorded here, and both trained models are in `models/`
(weights verified identical to their runs' `best.pt` before deletion). No
`last.pt` remains, so neither run can be resumed; retrain from `models/` or
the ADE20K checkpoints instead. All derived datasets (`vistas-yolo`,
`vistas-yolo-1280`, both `*-merge-v1`, `vistas8`) and the retention-study
JSONs were also deleted; only `training/datasets/vistas-raw` remains. The
original download archive (`vistas-v2.0-jxl.zip`, 21G) was deleted locally;
the user keeps a copy in Google Drive. Rebuild with the `vistas_convert.py` /
`vistas_merge.py` commands in the `fineTuningLimits` section below.

**Next:** label a test set from real production footage (both orientations,
~150-300 frames from distinct locations) before any more Vistas-only
training; then 300-500 labeled production frames for a fine-tune, oversampled
5-10x. Decide whether pedestrian-area can merge into sidewalk (it's the one
class the 544 fine-tune lost). Only real portrait footage can test the extra
near-ground field of view; the sky-padded simulation can't.

## HackEye frames dataset (2026-10-26): pseudo-labelled media library

**Status (2026-10-27): done, curated and renamed `HackEye26.1-124`.** 17,347 frames, both splits curated
with `fps2_sim0.90`: train 14,830 (from 169,696), val 2,517 (from 30,327); about 9 GB. See `datasets/hackeye/CURATION.md`. Next: snow-to-water
remap (suggested, not applied), Drive copy on the user's go, then retrain the app model on Vistas plus this set.
The plan and the scale study are in `datasets/hackeye/PLAN.md`;
location, layout and use are in `datasets/hackeye/README.md`.

- **What it is:** all 19 HackEye clips (`~/media/hackeye`) as 1280×720 q100 JPG frames at 20 fps,
  200,023 in total. Each frame has a 124-class mask from the X model. The dataset is at
  `~/media/hackeye-models/datasets/HackEye26.1-124` (`/datasets/HackEye26.1-124/HackEye26.1-124.yaml`). There is **no Drive copy
  yet**: on the user's go, the tar goes to `bpbroasGDrive:works/HackEye/datasets/`.
- **Split:** by clip. `val` = Chest-center-0002 and 0004, the Lab's test footage, 30,327 frames;
  `train` = the other 17 clips, 169,696 frames.
- **Extraction:** `extract_frames.py`, 6 clips in parallel, took about 20 min. All 200,023 expected
  frames were written, none were missing, and every sampled frame was 1280×720.
- **Scale:** the labels are the mean softmax of two inference scales, short sides 736 and 1280.
  On 500 Vistas val images degraded to 720p detail this scored mIoU 0.422, vs 0.402 at 736 alone
  and 0.424 at 1280 alone. It also scored best on the navigable surfaces and on pixel accuracy.
  Table in PLAN.md.
- **Labelling:** `pseudo_label.py` ran at batch 1 and about 5.1 frames/s, so about 11 h. Batch 1 is
  faster than batch 4 or 8 here, because the full-resolution softmax per image dominates. Peak VRAM
  is about 2.2 GiB. A Docker restart (NVIDIA driver 610 to 615 upgrade) stopped it at 116,760 frames on
  2026-10-26; it resumed after the reboot (the script skips finished frames) and finished on 2026-10-27
  at 03:30 UTC. Finalize passed.
- **Verify (2026-10-27):** 169,696 train and 30,327 val masks, one per frame. 600 sampled masks are `P` mode
  and every sampled frame is 1280×720. `vistas_verify.py yaml-check` passes (nc=124).
- **Teacher errors seen:** "nature snow" is breaking surf, sea foam and sun glare (21% of Chest-0006
  frames, 17.5% of Head-0004). "Water" (79% of frames) is mostly the real sea on the seafront clips.
- **Curation dry run:** `datasets/hackeye/curate.py` trims near-duplicates (a fixed fps subsample, then
  merging by 32×18 layout grid, with rare classes protected). It writes a report and keep-lists only. Results
  are in CURATION.md.
- **Gotchas found:**
  - A detached `docker compose run` needs `-T`, or its log output is held back. The extraction
    log stopped updating while the job kept running.
  - Ultralytics' predict stream holds a lock until the generator is exhausted. A half-consumed
    `stream=True` generator makes the next `predict()` call hang forever, which left one stuck
    smoke-test container. `pseudo_label.py` therefore drains it with `list(gen)`.

## X-1280 run (2026-10-18 to 10-26): dataset-creation model

**Result:** `models/Yolo26x-sem-Vistas124-HackEye-1280.pt`, in-training val **mIoU 0.4164,
pixel acc 0.9013** (124 classes, letterboxed to 1280; the best epoch was the last). For
reference, the old 124-class s@640 V1 run plateaued around 0.32 on the same metric. Its
purpose is dataset creation (curating and pre-labelling HackEye media-library frames), not
deployment, so it was never scored with the production-resolution eval.

- **Setup:** `yolo26x-sem-ade20k.pt` warm start, full 124-class Vistas at short side 1280
  (`vistas-yolo-1280`, rebuilt by `training/launch_x1280.sh`), imgsz 1280, batch 1,
  cls_pw 1.0, close_mosaic 3, cos_lr, workers 5, save_period 1. Run dir
  `/runs/vistas124-x-1280` (`~/media/hackeye-models/runs/`), container `hackeye-x1280`.
- **Speed:** about 2.19 h per epoch (2.3 it/s, 9.6–10 GB VRAM), far faster than the
  1.38 s/step probe predicted. A Lab GPU analysis running alongside drops it to about 1.5 it/s.
- **Curve:** mIoU 0.231 (ep1) -> 0.301 (11) -> 0.352 (20) -> 0.379 (30) -> 0.393 (40)
  -> 0.406 (50) -> 0.410 (60) -> 0.413 (70) -> 0.4159 (81). From epoch 50 the gains were
  about 0.0003 per epoch.
- **Skipped epochs 82–96 (user decision, 2026-10-25):** the gains no longer justified about
  33 h more. The container was stopped during epoch 82, `ckpt['epoch']` in `last.pt` was
  set to 95 (0-based), and the run was resumed with
  `train_semantic.py --data ... --resume /runs/vistas124-x-1280/weights/last.pt`. That made
  Ultralytics continue at epoch 97, with the cosine LR jumping to its end value (~0.0003)
  and mosaic closing for 98–100. `results.csv` jumps from epoch 81 to 97. Epochs 97–100
  were each a small new best (0.4160 -> 0.4164), with no dip from closing mosaic. The
  original epoch-81 checkpoint was kept as `weights/last_epoch81_backup.pt` until the teardown below.
- **Release:** the final `best.pt` (already stripped by Ultralytics: no optimizer or EMA,
  81 MB, fp32) was moved to `models/`; it loads as a 124-class semantic model. The
  per-epoch `epochN.pt` checkpoints were deleted in the teardown below.

## 720p scale check (2026-10-18)

Production is now 720p (1280x720; the net sees 1280x736), and the next model
is l. To decide its training scale, `models/Yolo26m-sem-HackEye26.1-48-1280.pt`
was scored on the full merged val (2000 imgs, rebuilt val-only
`vistas-yolo-1280{,-merge-v1}`, since deleted) with `diagnose_surfaces.py
--match-train-scale --imgsz {720,1280}`. The output (`/runs/scale-check/m1280-at-{720,1280}.json`) and the driver
(`/runs/scale720.sh`) were deleted on 2026-10-26; the numbers are below.

| | 1280 short side | 720 short side | delta |
|---|---|---|---|
| mIoU (47) / pixel acc | 0.567 / 0.910 | 0.548 / 0.902 | -0.019 / -0.008 |
| road / sidewalk / crosswalk / bike-lane | 0.868 / 0.671 / 0.680 / 0.442 | 0.857 / 0.676 / 0.666 / 0.444 | -0.01 to +0.005 |
| pedestrian-area / sidewalk-obstacle | 0.406 / 0.429 | 0.449 / 0.464 | +0.043 / +0.035 |
| curb / solid / dashed | 0.624 / 0.634 / 0.481 | 0.589 / 0.586 / 0.431 | -0.035 to -0.050 |
| pole / ped light / person | 0.608 / 0.589 / 0.766 | 0.540 / 0.533 / 0.716 | -0.05 to -0.07 |

The 1280 row reproduces the 2026-10-12 table. The overall drop is about half
of the drop at 540 (-0.041), with the same pattern: thin cues and small
objects lose, broad surfaces hold or gain (pedestrian-area and
sidewalk-obstacle improve at 720). The drop mixes lost detail (unavoidable
at 720) with train/test scale mismatch (recoverable by training at 720).
**Decision input:** -0.019 sits between "safe to train at 1280" (<=0.01) and
"train at 736" (~0.03). Recommended: train l at `--imgsz 736` with the usual
+-0.5 scale aug. **Baseline to beat at 720: mIoU 0.548**, plus the thin-cue
rows above.

## Production scale (540) check and 544 fine-tune (2026-10-12)

Production is expected at half 1080p (960x540; the net sees 960x544). Scored
`models/Yolo26m-sem-HackEye26.1-48-1280.pt` on the full merged val (2000 imgs,
rebuilt val-only `vistas-yolo-1280{,-merge-v1}`) with `diagnose_surfaces.py
--match-train-scale --imgsz {540,1280}`; output `/runs/scale-check/`.

| | 1280 short side | 540 short side | delta |
|---|---|---|---|
| mIoU (47) / pixel acc | 0.567 / 0.910 | 0.526 / 0.891 | -0.041 / -0.019 |
| road / sidewalk / crosswalk | 0.868 / 0.671 / 0.680 | 0.845 / 0.661 / 0.653 | -0.02 to -0.03 |
| bike-lane / ped-area | 0.442 / 0.406 | 0.406 / 0.411 | -0.036 / +0.006 |
| curb / solid / dashed | 0.624 / 0.634 / 0.481 | 0.553 / 0.541 / 0.389 | -0.07 to -0.09 |
| pole / ped light / person | 0.608 / 0.589 / 0.766 | 0.484 / 0.466 / 0.667 | -0.10 to -0.12 |

Broad surfaces barely move; thin cues and small objects lose most of what
1280 gained. Surface errors grow at boundaries (road 0-3px error 51% -> 68%),
interiors unchanged. Training saw short sides ~640-1920 (resize_short to
1280, scale +-0.5), so 540 is below its range. Next: short fine-tune from
this model at `--imgsz 544`.

**Launched 2026-10-12** as detached container `hackeye-m544ft` (after
rebuilding `vistas-yolo` 768 + `vistas-yolo-merge-v1` from raw with the
commands in the section below):

```bash
docker compose run -d --name hackeye-m544ft train training/train_semantic.py \
  --model /app/models/Yolo26m-sem-HackEye26.1-48-1280.pt --data /datasets/vistas-yolo-merge-v1/vistas-v2.0.yaml \
  --imgsz 544 --batch 8 --cls-pw 1.0 --workers 5 --no-plots --epochs 20 --close-mosaic 3 \
  --optimizer SGD --lr0 0.002 --save-period 5 --name vistas48-m-544ft
```

Evaluate the result with the same `diagnose_surfaces.py --match-train-scale
--imgsz 540` command so it compares directly with the table above.

**Mid-run checks at 540** (`training/build_orientation_val.py` builds
`/datasets/val540-orientation/{landscape,center,portrait}` from the 1280
merged val; scored on `landscape` = full frame at 540 short side, GT at 540,
so the original scores 0.5235 here vs 0.526 against 1280 GT above).
Checkpoint files are 0-based: `epoch5.pt` = end of epoch 6, `epoch10.pt` =
end of epoch 11. Run at SGD 0.002, 3 warmup epochs, ~750 s/epoch, batch 8
(5.4 GB), in-training val mIoU 0.487-0.495 throughout.

| | original | ep6 | ep11 | ep11 vs orig |
|---|---|---|---|---|
| mIoU (47) / pixel acc | 0.5235 / 0.889 | 0.5078 / 0.890 | 0.5187 / 0.893 | -0.005 / +0.004 |
| road / sidewalk / crosswalk | 0.842 / 0.657 / 0.649 | 0.847 / 0.630 / 0.649 | 0.856 / 0.645 / 0.649 | +0.013 / -0.012 / +0.001 |
| bike-lane / ped-area | 0.411 / 0.413 | 0.413 / 0.373 | 0.443 / 0.323 | +0.032 / -0.091 |
| curb / solid / dashed | 0.549 / 0.526 / 0.382 | 0.545 / 0.540 / 0.404 | 0.554 / 0.560 / 0.431 | +0.005 / +0.034 / +0.049 |
| person / pole / ped light | 0.656 / 0.474 / 0.461 | 0.649 / 0.476 / 0.450 | 0.655 / 0.476 / 0.449 | -0.001 / +0.003 / -0.012 |
| classes better than original | | 24/47 | 30/47 | |

At ep11 the pre-agreed rule (still below original mIoU -> stop, purge,
restart at lr0 0.0004) technically triggered, but the user chose to let it
finish: mIoU was climbing (+0.011 over 5 epochs) with most of the LR decay
ahead, and the gap is almost all pedestrian-area (likely sidewalk ambiguity,
not LR). In-training val (letterboxed to long side 544, smaller than
production) does not track the 540 eval: `best.pt` it picked is epoch 2, so
evaluate `last.pt`, not `best.pt`.

**Final result.** 20 epochs, ~4h10m, exited cleanly (in-training val mIoU
0.492 at epoch 20; `best.pt` picked epoch 2 by that small-scale val, ignore
it). `last.pt` (already optimizer-stripped, 29M) was committed as
`models/Yolo26m-sem-HackEye20261012-544.pt` (removed 2026-10-18) (sha256 e6550116...). Scored with
`diagnose_surfaces.py --match-train-scale --imgsz 540` on the three
`val540-orientation` variants; the portrait pad rows are mask 255 (ignored).
Output in `/runs/orientation-eval/{orig,ft}-{landscape,center,portrait}`.

| orig -> ft | landscape 960x540 | center 540x540 | portrait 540x960 |
|---|---|---|---|
| mIoU (47) | 0.524 -> 0.525 | 0.508 -> 0.516 | 0.514 -> 0.513 |
| pixel acc | 0.889 -> 0.895 | 0.891 -> 0.897 | 0.891 -> 0.896 |
| road | +0.016 | +0.014 | +0.012 |
| sidewalk | -0.004 | +0.009 | -0.002 |
| crosswalk | +0.004 | +0.010 | 0.000 |
| bike-lane | +0.043 | +0.042 | +0.017 |
| pedestrian-area | -0.027 | -0.022 | -0.044 |
| curb | +0.004 | +0.007 | +0.005 |
| solid / dashed line | +0.030 / +0.034 | +0.028 / +0.037 | +0.031 / +0.038 |
| person / pole | +0.002 / +0.002 | +0.003 / +0.003 | +0.006 / +0.019 |
| bicycle symbol | +0.032 | +0.068 | +0.038 |
| classes better | 35/47 | 38/47 | 33/47 |

- Fine-tune: mIoU about level, but pixel acc up everywhere and the
  navigation classes gain (road, bike-lane, lane lines, bicycle symbol, curb
  slightly). Only real loss: pedestrian-area, mostly to sidewalk.
- Portrait: center vs portrait (same pixels plus the sky band) moves mIoU by
  <0.01 for either model, so a tall frame with the content at the bottom is
  fine. Center/landscape aren't directly comparable (the crop drops side
  classes and context).
- For next runs: the in-training val (letterboxed to long side 544) doesn't
  track production-scale results. Judge checkpoints with the 540 eval.
  The 3 warmup epochs at lr0 0.002 cost ~0.016 mIoU that took until about
  epoch 15 to recover; a lower lr0 / shorter warmup is worth trying.

## Branch `fineTuningLimits` (2026-10-06): merged taxonomy + m@1280 run

**Diagnostic** (`training/diagnose_surfaces.py`, output
`/runs/diag-surfaces/val_full.{json,txt}`): bike-lane, crosswalk-plain and
pedestrian-area fail as whole regions (error ~35-48% even 96+px from any
boundary, flipping to road/sidewalk), i.e. region/context failure, not
boundary localization.

**Class merge HackEye26.1-48** (formerly "merge v1"; `datasets/vistas/conversion/merges/HackEye26.1-48.yaml`): 124 -> 48
classes (47 trained, only `void unlabeled` ignored). Datasets are derived,
never hand-edited:

```bash
# 768 short side (from the existing conversion)
docker compose run --rm --entrypoint python train datasets/vistas/conversion/vistas_merge.py \
  --src /datasets/vistas-yolo --dst /datasets/vistas-yolo-merge-v1 \
  --map datasets/vistas/conversion/merges/HackEye26.1-48.yaml --config /datasets/vistas-raw/config_v2.0.json

# 1280 short side: re-convert from raw (upscaling the 768 set would add no detail),
# long-side cap raised so 16:9 frames aren't shrunk below 1280 short side
docker compose run --rm --entrypoint python train datasets/vistas/conversion/vistas_convert.py \
  --src /datasets/vistas-raw --dst /datasets/vistas-yolo-1280 --short-side 1280 --max-long-side 4096 --workers 12
docker compose run --rm --entrypoint python train datasets/vistas/conversion/vistas_merge.py \
  --src /datasets/vistas-yolo-1280 --dst /datasets/vistas-yolo-1280-merge-v1 \
  --map datasets/vistas/conversion/merges/HackEye26.1-48.yaml --config /datasets/vistas-raw/config_v2.0.json
```

Both merged sets: 20000/20000 masks, pixel accounting OK, `vistas_verify.py
shapes`/`yaml-check` clean (nc=48). 1280 conversion is 11G (+537M merged
masks, images hardlinked).

**Timing probes** (60 iters, cls_pw 1.0, 5 workers, train only, no val):

| model | imgsz | batch | s/iter | img/s | train min/epoch | peak reserved |
|---|---|---|---|---|---|---|
| l | 640 | 8 | 1.19 | 6.75 | 44 | 16.0 GB |
| l | 1280 | 2 | 1.27 | 1.57 | 191 | 16.3 GB |
| l | 1280 | 4 | 1.27 | 3.16 | 95 | 16.3 GB |
| m | 1280 | 4 | 1.26 | 3.17 | 95 | 16.2 GB |
| m | 1280 | 6 | 1.92 | 3.13 | 96 | 16.4 GB |

Throughput flat across batch at 1280 because the GPU saturates even at small
batch (verified on the live run: GPU 99-100%, the 5 train dataloader workers
~2% CPU each, load avg ~1.1 on 12 cores) -- GPU-bound, not dataloader-bound,
so neither a bigger batch nor more workers speeds it up. Gradient quality
doesn't depend on batch either: Ultralytics accumulates to `nbs=64`; only
per-step BatchNorm stats differ. Measured epoch time on the real run is
~70 min incl. val (probe's 95 min was inflated by the concurrent CPU-heavy
1280 re-convert).

**The run is actually at batch 2, not 4.** 200 iterations into epoch 1 it
hit a real CUDA OOM at batch 4 and Ultralytics auto-retried at batch 2
(log: `CUDA out of memory with batch=4. Reducing to batch=2 and retrying
(1/3)`; 9000 iters/epoch, ~7.7G). `args.yaml` still says `batch: 4` -- check
the log, not args.yaml, for the effective batch. The 30-iter probe's batch 4
passed only because it was too short to hit the peak. Same speed either way
(GPU-bound); cost is noisier per-step BatchNorm stats. A `--resume` will try
4 again and should fall back to 2 the same way.

**Ran 2026-10-06 -> 2026-10-11** as detached container `hackeye-m1280` (since removed):

```bash
docker compose run -d --name hackeye-m1280 train training/train_semantic.py \
  --model /app/models/yolo26m-sem-ade20k.pt --data /datasets/vistas-yolo-1280-merge-v1/vistas-v2.0.yaml \
  --imgsz 1280 --batch 4 --cls-pw 1.0 --workers 5 --no-plots --epochs 100 --name vistas48-m-1280
# (effective batch is 2 -- auto-reduced after an OOM, see above)
# resume: docker rm hackeye-m1280 && docker compose run -d --name hackeye-m1280 train \
#   training/train_semantic.py --resume /runs/vistas48-m-1280/weights/last.pt
```

**Mid-run comparison (epoch 40 snapshot vs old final s@640)**, CPU-only so
training was untouched (`diagnose_surfaces.py --device cpu --match-train-scale`,
old model via `--pred-remap .../merge_manifest.json`, both scored on the same
1280 merged val GT; output in `/runs/comparisons/compare-ep40/`):

| | old s@640 final | new m@1280 ep40 |
|---|---|---|
| mIoU (47 cls) | 0.524 | 0.552 |
| pixel acc | 0.907 | 0.903 |
| road / sidewalk | 0.874 / 0.662 | 0.849 / 0.662 |
| crosswalk / bike-lane / ped-area | 0.671 / 0.426 / 0.416 | 0.632 / 0.404 / 0.412 |
| curb / solid / dashed | 0.575 / 0.590 / 0.424 | 0.616 / 0.625 / 0.467 |
| bicycle symbol / ped light / person / pole | 0.451 / 0.404 / 0.680 / 0.520 | 0.534 / 0.557 / 0.750 / 0.591 |

30/47 classes better. Thin cues/objects gain (resolution); broad surfaces
slightly worse. Whole-region failures dropped (bike-lane 33.3% -> 17.2%,
crosswalk 13.7 -> 9.9, road-shoulder 46.4 -> 30.8; ped-area barely, 46.3 ->
40.7) while road's rose 2.2 -> 5.8%: the new model over-predicts rare surfaces
over road, consistent with `cls_pw=1.0`. Caveat: epoch 40/100 at LR ~0.020 vs
a finished model (the old run gained only +0.011 mIoU from epoch 40 to 100).

**Final result (2026-10-11).** Training stopped at 04:28 UTC during epoch
94 with `CUDA error: an illegal memory access` in the CE loss; kernel log
showed NVIDIA Xid 31 (GPU MMU fault) at the same second -- first crash in
~4.5 days, GPU healthy afterwards. Not resumed: the remaining 7 epochs were
all in the no-mosaic phase, where mIoU was slipping ~0.0001/epoch
(0.5528 @89 -> 0.5525 @93), same as the 124-class run's no-mosaic phase
(0.3196 -> 0.3186). `last.pt` = epoch 93 if a resume is ever wanted.

- **Final model:** `best.pt` = epoch 89, val mIoU 0.5528, pixel acc 0.9055,
  stripped (`strip_optimizer`, 86M -> 29M, all 424 tensors and sample
  predictions bit-identical) and committed as
  `models/Yolo26m-sem-HackEye26.1-48-1280.pt`. Full checkpoint (with optimizer)
  stays at `/runs/vistas48-m-1280/weights/best.pt`.
- **Curve:** mIoU 0.398 (ep1) -> 0.460 (10) -> 0.513 (20) -> 0.527 (30) ->
  0.542 (40) -> 0.549 (50) -> 0.551 (60-75 plateau) -> 0.553 (89). Val CE
  bottomed ~0.539 at epoch 44 and rose to ~0.605 by 93 while train CE kept
  falling -- mild overconfidence, never cost mIoU; ~epoch 80 would have
  given nearly the same model. For next runs: `close_mosaic` 0-5 or stop at
  plateau; the no-mosaic phase didn't help either run.
- **Comparison vs old 124-class s@640** (epoch-64 snapshot, same 1280 merged
  val GT, `/runs/comparisons/compare-ep65/`): mIoU 0.524 -> 0.563, pixel acc
  0.907 -> 0.908, 40/47 classes better. Boundary cues and thin objects
  clearly up (curb +0.047, lane lines ~+0.05, bicycle symbol +0.10,
  pedestrian light +0.175, pole +0.08, person +0.08); broad surfaces about
  level on IoU but with a different error profile: bike-lane recall 60% ->
  80% (bike-lane->road 28% -> 8%), crosswalk 78% -> 87%, at the cost of road
  being over-predicted as bike-lane/crosswalk (road recall 95% -> 89%) --
  the `cls_pw=1.0` trade-off. Slightly worse: road-shoulder -0.027,
  service-lane -0.019, road -0.013. Still failing: pothole, animal (too
  little data); pedestrian-area vs sidewalk confusion (~29%) may be label
  ambiguity.

`yolo26m-sem-ade20k.pt` is an official asset
(`github.com/ultralytics/assets/releases/download/v8.4.0/`), cached in
`models/`. Transfer 420/424 (deficit 4, as expected). Caveat: production
inference is planned below 640 and rectangular, so a 1280-trained model
will see a train/test scale mismatch there — evaluate at the export shape.

**Repo reorganized into folders 2026-10-03** (see PLAN.md's Layout section)
— scripts moved under `docker/`, `training/`, `datasets/vistas/conversion/`,
`models/`. Commands below predating that date still show the old flat
paths (accurate history, left as-is); new commands use the new paths. Run
`docker compose` from the `docker/` directory (or pass
`-f docker/compose.yaml` from the repo root) — container WORKDIR/mount is
now `/app`, not `/training`. This did **not** touch the live training
container (already running, keeps its files loaded in memory regardless of
on-disk moves) or the external `/datasets`/`/runs` mounts, which are
staying external until this run finishes (deferred, see TODO at the bottom
of this file).

## FINAL EVALUATION RESULTS (2026-10-03)

Ran against `best.pt` (which, by end of training, coincided with the final
epoch 100 checkpoint — `best.pt`/`last.pt` are byte-identical, same
timestamp, since mIoU peaked right at the end):

```bash
docker compose run --rm train training/eval_semantic.py \
  --weights /runs/vistas124-s-640/weights/best.pt \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --target-keywords "flat road,flat sidewalk,bike-lane,pedestrian-area,crosswalk-plain,marking--discrete--crosswalk-zebra,flat parking,service-lane,rail-track,traffic-island,curb" \
  --out-csv /runs/eval-final/full_124class_summary.csv \
  --project /runs --name eval-final
```

**Headline**: mIoU 0.3197, pixel_acc 0.8957 (116/124 classes present in val
GT). Matches PLAN.md's calibration note (high-20s to mid-30s expected for a
~10M-param model) — not the real acceptance criterion, see below.

**Priority-1 classes** (the actual acceptance criterion, per
[[priority-broad-navigable-classes]]), sorted by IoU:

| class | IoU | pixel_acc |
|---|---|---|
| flat road | 0.867 | 0.942 |
| marking--discrete--crosswalk-zebra | 0.700 | 0.824 |
| flat sidewalk | 0.662 | 0.808 |
| flat rail-track | 0.498 | 0.727 |
| flat parking-aisle | 0.471 | 0.519 |
| flat service-lane | 0.456 | 0.660 |
| flat pedestrian-area | 0.433 | 0.528 |
| flat crosswalk-plain | 0.424 | 0.591 |
| flat bike-lane | 0.400 | 0.586 |
| flat traffic-island | 0.359 | 0.468 |
| flat road-shoulder | 0.289 | 0.464 |
| flat parking | 0.207 | 0.279 |

**Average IoU across these 12 classes: ~0.480** — well above the headline
0.32, confirming the headline number was being dragged down by the long
tail of rare classes, not the ones that matter for this project. Road and
sidewalk, the two most safety-critical for navigation, are the strongest
performers.

**Curb/curb-cut** (deliberately deprioritized this round, reference only):
`barrier curb` 0.538 IoU, `flat curb-cut` 0.204 IoU — meaningfully weaker
than the priority classes, confirms the need for the already-planned
dedicated fine-tuning pass.

**Confusion matrix findings worth keeping:**
- `flat road` is very clean: 96.6% of its GT pixels predicted correctly.
- `flat parking` (weakest priority class, 0.207 IoU) is predicted as
  `flat road` 54.3% of the time; `flat parking-aisle` similarly 32.5% as
  road. The model is treating parking areas as generic driveable surface
  rather than a distinct category — not surprising given visual similarity,
  worth knowing if parking-area precision ever matters downstream.
- `barrier curb`: 75.7% correct, but 11.5% misclassified as road and 7.0%
  as sidewalk — the classic boundary-confusion pattern. `eval_semantic.py`'s
  own built-in note on this: "if curb -> sidewalk/road dominates, the model
  sees the boundary region but can't localize the transition -- that argues
  for higher --imgsz, not more epochs." Directly supports the already-saved
  higher-res-crop-finetuning-for-curb memory/plan for the next phase.
- Full 124-class CSV saved at `/runs/eval-final/full_124class_summary.csv`
  and prediction/label overlay images at `/runs/eval-final/results/` for
  anyone wanting to look beyond this summary.

## If it crashes / needs to stop: how to resume

`training/train_semantic.py` has a `--resume` flag (added 2026-10-01, before
starting the full run, specifically so a crash/power-loss/manual stop on a
~40h run doesn't lose real progress). To resume, from the `docker/` directory:

```bash
docker compose run --rm train training/train_semantic.py \
  --resume /runs/vistas124-s-640/weights/last.pt
```

That's the *entire* command — `--resume` ignores all other flags and
restores the original run's full config (data, epochs, cls_pw, etc.) from
the checkpoint's own saved args, continuing from the last completed epoch
with optimizer/scheduler/EMA state intact (verified against Ultralytics'
own `check_resume()`/`resume_training()` source, not just the docs).

`last.pt` is overwritten every epoch regardless of `--save-period` (only
`best.pt` and the periodic numbered snapshots depend on that setting), so
**at most one epoch's progress (~20-25 min at this run's pace) is ever at
risk** from an unclean stop. To deliberately pause: `docker stop
<container-name>` (or `Ctrl-C` on a foreground run) is safe at any point —
the resume command above picks it back up regardless of exactly when it
stopped.

## FINAL verdict: ADE20K init + cls_pw=0.0

| class | ADE20K, cls_pw=0 | ADE20K, cls_pw=1 | Cityscapes, cls_pw=0 |
|---|---|---|---|
| flat road | **0.836** | 0.783 | 0.825 |
| flat sidewalk | **0.604** | 0.574 | 0.570 |
| crosswalk-zebra | **0.579** | 0.523 | 0.545 |
| flat pedestrian-area | **0.475** | 0.369 | 0.363 |
| flat service-lane | **0.414** | 0.381 | 0.379 |
| flat bike-lane | **0.375** | 0.338 | 0.298 |
| flat rail-track | **0.372** | 0.346 | 0.238 |
| flat traffic-island | **0.235** | 0.232 | 0.195 |
| flat road-shoulder | 0.154 | **0.171** | 0.152 |
| flat crosswalk-plain | 0.145 | **0.221** | 0.020 |
| flat parking | 0.093 | **0.152** | 0.044 |
| flat parking-aisle | 0.0 | **0.162** | 0.0 |
| overall mIoU | 0.233 | **0.236** | 0.187 |
| overall pixel_acc | **0.876** | 0.846 | 0.870 |

Config 4 (Cityscapes, `cls_pw=1.0`) was stopped ~5 minutes in, deliberately
— **user's call**: Cityscapes' fundamental limitation is its narrow
19-class taxonomy, which makes any checkpoint pretrained on it too
specialized/biased for a 124-class target taxonomy as diverse as Vistas',
regardless of `cls_pw`. Three configs' worth of evidence (Cityscapes
trailing ADE20K by wide margins on nearly every priority class at
`cls_pw=0`) was judged sufficient to rule it out without needing to see
`cls_pw=1.0` applied to the same weak checkpoint.

**Decided: ADE20K init + `cls_pw=0.0`** (config 1) — best or tied-best on
8 of 12 priority classes, including the largest margins on road, sidewalk,
and pedestrian-area, plus the best overall pixel accuracy.

## Crash root cause: CONFIRMED FIXED

Config 2 (`ab-ade20k-clspw1`) crashed twice at `--workers 6/8` with default
plotting on, then **completed cleanly** on a third attempt at
`--workers 5 --no-plots`, running well past the wall-clock point (epoch
8's validation) where it previously died, with swap staying completely flat
(908MB, unchanged) the entire ~4.3h run. This confirms the matplotlib
plot-leak hypothesis (or at minimum, that `--no-plots` + `workers=5`
together resolve it) well enough to trust for the remaining A/B configs.
**Use `--batch 8 --workers 5 --no-plots` for configs 3-4 and revisit
whether the full 100-epoch run needs plots badly enough to risk re-enabling
them (if so, test that in isolation first rather than assuming it's safe).**

Before each config launch: explicitly verify `docker ps` is empty, GPU is
at 0MB, and swap is at baseline before starting the next one — cheap
insurance against any cross-run contamination, even though each config
already runs in its own fresh `docker compose run --rm` container.

## cls_pw verdict (ADE20K checkpoint) — cls_pw=0.0 wins for this project's priority

| class | cls_pw=0 | cls_pw=1 | diff |
|---|---|---|---|
| flat road | 0.836 | 0.783 | −0.053 |
| flat sidewalk | 0.604 | 0.574 | −0.030 |
| crosswalk-zebra | 0.579 | 0.523 | −0.056 |
| flat pedestrian-area | 0.475 | 0.369 | −0.106 |
| flat service-lane | 0.414 | 0.381 | −0.033 |
| flat bike-lane | 0.375 | 0.338 | −0.037 |
| flat rail-track | 0.372 | 0.346 | −0.026 |
| flat traffic-island | 0.235 | 0.232 | ~tied |
| flat road-shoulder | 0.154 | 0.171 | +0.017 |
| flat crosswalk-plain | 0.145 | 0.221 | +0.076 |
| flat parking | 0.093 | 0.152 | +0.059 |
| flat parking-aisle | 0.0 | 0.162 | +0.162 |
| overall mIoU | 0.233 | 0.236 | ~tied |
| overall pixel_acc | 0.876 | 0.846 | −0.030 |

`cls_pw=1.0` wins on overall mIoU and on several rare classes (parking,
parking-aisle, crosswalk-plain) but loses on nearly every priority-1 class
(road, sidewalk, pedestrian-area, bike-lane, service-lane, rail-track).
Given the project's stated priority (broad navigable classes over rare
ones), `cls_pw=0.0` is the clear choice **for the ADE20K checkpoint**. Still
need the Cityscapes-init comparison (configs 3-4) to see if this holds for
that checkpoint too, and to settle the checkpoint question itself.

## A/B priority reframing (see memory: priority-broad-navigable-classes)

Per explicit user direction 2026-09-30, this A/B round (and the eventual
full run) is judged on IoU for broad navigable classes (`road`, `sidewalk`,
`bike-lane`, `pedestrian-area`, `crosswalk-plain`/`crosswalk-zebra`,
`parking`/`parking-aisle`, `road-shoulder`, `service-lane`, `rail-track`,
`traffic-island`) — **not** curb/curb-cut, which is a deliberately deferred
later fine-tuning pass. PLAN.md step 5's text is stale on this point; treat
this file and the saved memory as authoritative.

## A/B results so far

**Config 1/4 — `ab-ade20k-clspw0`** (ADE20K init, `cls_pw=0.0`, batch=8,
workers=6, 10 epochs, full dataset): **completed cleanly.**
Overall mIoU 0.233, pixel_acc 0.876.

| class | IoU | pixel acc |
|---|---|---|
| flat road | 0.836 | 0.935 |
| flat sidewalk | 0.604 | 0.783 |
| crosswalk-zebra | 0.579 | 0.739 |
| flat pedestrian-area | 0.475 | 0.649 |
| flat service-lane | 0.414 | 0.598 |
| flat bike-lane | 0.375 | 0.558 |
| flat rail-track | 0.372 | 0.511 |
| flat traffic-island | 0.235 | 0.299 |
| flat road-shoulder | 0.154 | 0.214 |
| flat crosswalk-plain | 0.145 | 0.182 |
| flat parking | 0.093 | 0.111 |
| flat parking-aisle | 0.0 | 0.0 (only 25 val images, low support) |

**Config 2/4 — `ab-ade20k-clspw1`**: crashed twice, see below. Partial
epoch-level trend before the 2nd crash (7/10 epochs, no final per-class
breakdown yet): mIoU 0.123 → 0.164 → 0.189 → 0.205 → 0.218 → 0.228 → 0.232 —
already close to config 1's full-10-epoch final mIoU after only 7 epochs,
interesting but not conclusive without the completed run.

**Configs 3-4** (`yolo26s-sem.pt` / Cityscapes init, both `cls_pw` values):
not started yet.

## Two real training crashes investigated — root causes and fixes

### Crash 1 (first attempt at config 1, `--workers 8`, batch=16 default)

Two-stage failure: (a) `batch=16` hit a real CUDA OOM, Ultralytics
auto-retried at `batch=8` — already covered above (`train_semantic.py`
`--batch` default fixed to 8). (b) Separately, the orchestration **wrapper
script** sequencing all 4 A/B runs got killed by the harness
("system is running low on memory") while the actual training container
kept running fine underneath it — confirmed via `docker ps`
(`Up N minutes`, still progressing) even after the wrapper died. Not a real
training failure, just lost the auto-sequencing of configs 2-4. Fixed by
switching to launching each config as its own standalone background task
instead of one long multi-hour wrapper script, so losing one wrapper
can't silently kill the whole chain.

### Crash 2 (config 1 retry, `--workers 8` default still in effect)

This one was real: the training container itself died, not just a wrapper.
Investigated with `dmesg`/`journalctl` — **no kernel OOM-killer entry**, but
`systemd` logged the container's cgroup scope deactivating after
**4h14m wall-clock, 13.8G memory peak**, while the training log showed
almost no progress for the last ~3 hours (frozen at epoch 4, iteration
40/2250) — i.e. the process was alive but thrashing on swap, not quickly
OOM-killed. `ps aux` at the time showed **21 processes** for a
`--workers 8` config, way more than expected. Fixed (per explicit user
choice) by lowering `train_semantic.py`'s `--workers` default from 8 to 4,
on the theory that fewer persistent DataLoader workers (~2.5GB RSS each,
a known PyTorch multiprocessing copy-on-write pitfall) would reduce
baseline memory pressure.

**Important correction after reading Ultralytics' source**: its
`InfiniteDataLoader` (`ultralytics/data/build.py`) is explicitly designed to
*reuse* persistent workers across epochs, not recreate them — so "workers
leaking per-epoch" is not the precise mechanism. The 21-process count and
the slow multi-hour memory growth have a different, still only
partially-understood cause.

### Crash 3 (config 1 clean restart at `--workers 6`, user's explicit choice)

**Succeeded** — completed all 10 epochs cleanly (results above). This was
the first fully clean run, consistent with `--workers 6` giving enough
margin, OR just being lucky that config 1 happens to not trigger whatever
the growth source is within one run's duration.

### Crash 4 (config 2, same `--workers 6`, otherwise identical setup)

**Crashed again** — but differently from crash 2: died near the *end* of
the run (4h17m wall-clock, stuck in what looks like the final epoch's
validation, 48% through, "13.5G memory peak" per systemd) rather than
early-and-thrashing. Memory was fully back to normal (1.9GB used, 28GB
available) within minutes of the kill, confirming it's this process's own
growth, not unrelated host pressure. No kernel OOM entry this time either.

**New hypothesis, not yet confirmed**: Ultralytics' plotting code
(`ultralytics/utils/plotting.py`, verified by reading the source) defines 6
plot functions but calls `plt.close()` in only 4 of them — a known
matplotlib-figure-leak pattern (unclosed figures accumulate in matplotlib's
global figure registry) that would compound specifically over *many hours
of a long run*, matching the "dies near the end, not early" symptom better
than a per-worker-count theory does. `train_semantic.py` hardcoded
`plots=True`. **Fix applied**: added `--plots`/`--no-plots` CLI flag
(default stays `True` for the eventual full run, where plots are worth
reviewing), relaunched config 2 with `--no-plots`. **Not yet confirmed** —
if this run also crashes, the plots theory is wrong and the real cause is
still open. If it completes cleanly, use `--no-plots` for the remaining A/B
configs too, and reconsider whether the full 100-epoch run should also use
it (losing the plots would be a real cost there, so a full fix — properly
closing figures, or periodically calling `plt.close('all')` — would be
better than permanently disabling them if this theory confirms out).

## Watch for next session

If crashes keep happening even with `--no-plots`, worth trying: `--workers 2`
(more conservative than 6), explicit `del`/`gc.collect()` calls around the
Ultralytics validator between epochs (can't change library code easily from
here, but could monkeypatch or file an upstream issue), or just accepting
checkpointed/resumed runs as normal operating procedure for this box rather
than expecting a single unattended multi-hour run to always complete.

## Batch size: original plan's estimate was wrong, fixed 2026-09-30

First real A/B run (`ab-ade20k-clspw0`, full dataset) hit a real `CUDA out
of memory` at the then-default `batch=16` and Ultralytics auto-retried at
`batch=8` — which worked, but wasn't the deliberate choice train_semantic.py
was using, and cost real time re-discovering. Stopped it after ~1h (was 37%
into epoch 3/10) to fix properly rather than let all 4 A/B configs pay the
same OOM-retry tax.

Tried Ultralytics' own `AutoBatch` profiler (`--batch -1`) to find a good
number automatically instead of guessing — it was **also wrong**: estimated
batch=30 was safe (59% GPU mem) from a synthetic forward/backward profile,
then itself hit real OOMs at 30 and at 15 before settling on 7. AutoBatch's
synthetic profile apparently doesn't capture the real augmentation pipeline,
aux head, dice loss, or other real-training-step overhead closely enough to
be trusted here.

**Fix:** `train_semantic.py --batch` default changed from 16 → **8**, the
number already proven stable across multiple real epochs on the full
18,000-image dataset (12.1GB / 15.9GB, comfortable headroom). See
`PLAN.md`'s VRAM/wall-clock sections (also corrected) for the numbers.

**Consequence — this pushes the full 100-epoch run's estimated wall-clock
from ~6-9h to ~38-42h**, since throughput is roughly halved from the
original batch=16 assumption. Flagged to the user 2026-09-30: decided to
accept this and proceed as-is (A/B restarted at `batch=8`, ~15-16h
background run) rather than delay to chase more speed first. Revisit
full-run wall-clock (imgsz, gradient accumulation, etc.) after A/B results
are in, if it still matters at that point.

## Environment

- Host: single NVIDIA RTX 5060 Ti, 16GB VRAM, Blackwell (sm_120).
- Docker image `hackeye-train:latest` — build from this repo:
  `docker compose build`. Rebuilt and verified working from this repo's new
  location (all five scripts import cleanly inside the container).
- No `unzip`/`7z` on host or in the image — only Python's stdlib `zipfile`
  (works fine, just slower than native tools; extraction of the full archive
  took ~80s for 165k files).
- **Run everything as `--user 1000:1000` (or via `docker compose`, which
  already sets this)** — an earlier round of ad-hoc `docker run` testing
  without a user flag left root-owned files that were a pain to clean up.
  `compose.yaml` already has `user: "${UID:-1000}:${GID:-1000}"` baked in.

## Data — all done, verified

| What | Where | Status |
|---|---|---|
| Original archive | `/home/megabeast/datasets/mapilliaryVistasV2.zip` (29GB) | Downloaded, moved here as a backup. Safe to delete once confident (raw extraction below is verified against it). |
| Raw extracted | `/home/megabeast/datasets/vistas-raw/` (31GB) | Extracted, verified: 18,000 train / 2,000 val / 5,000 test images+labels, matches Vistas v2.0 spec exactly. `config_v2.0.json` confirmed: **124 classes, 8 void (`evaluate: false`) classes**, label PNGs mode `P` as expected. |
| Converted (YOLO format) | `/home/megabeast/datasets/vistas-yolo/` (4.3GB) | Full conversion run: `vistas_convert.py --src /datasets/vistas-raw --dst /datasets/vistas-yolo --short-side 768 --max-long-side 2048 --workers 12`. **18,000/18,000 train converted, 0 errors. 2,000/2,000 val converted, 0 errors.** `vistas-v2.0.yaml` + `conversion_manifest.json` written. |
| Runs dir | `/home/megabeast/runs` | **Not yet created** — `compose.yaml` defaults `HACKEYE_RUNS_DIR` to this path; `mkdir -p` it before the first real training run (Docker will auto-create it on mount, but as root unless you mkdir first as yourself). |

### Real taxonomy surprises worth knowing (both harmless — code derives everything from the config, nothing hardcoded)

- `curb` is `construction--barrier--curb`, **not** `construction--flat--curb`
  as assumed earlier in the design process. `curb-cut` is a different
  branch: `construction--flat--curb-cut`. They are not siblings.
- 8 void/non-evaluate classes in the real config, not the 1-2 originally
  guessed.

### Retention study result (already recorded in `PLAN.md`, repeated here since it changed the plan)

Ran `vistas_verify.py retention` against real data, sweeping short-sides from
192 to 1536. **No retention cliff anywhere in that range** — `barrier curb`
and `flat curb-cut` actually *gain* relative pixel share under downsampling
(Vistas draws curb polygons wide enough to survive aggressive
nearest-neighbour resizing). This means `--short-side` is a
speed/localization-precision knob here, not a safety gate. Went with the
plan's original default of 768 since nothing argued against it.

## Verification — all pre-training gates done

- ✅ `vistas_verify.py shapes --dst /datasets/vistas-yolo` — **20,000/20,000
  pairs, 0 failures**, run against the full real converted dataset.
- ✅ `vistas_verify.py yaml-check --yaml /datasets/vistas-yolo/vistas-v2.0.yaml`
  — exit 0, `masks_dir='masks'`, 124 names, `check_det_dataset: nc=124`, no
  `PROBLEM:` lines (selects `SemanticDataset`, not `PolygonSemanticDataset`).
- ✅ `vistas_verify.py roundtrip --src /datasets/vistas-raw --dst
  /datasets/vistas-yolo --n 200` — **400/400 pairs checked (200 train + 200
  val), 0 failures.**
- ✅ `vistas_verify.py overlay` — 12 overlays written to
  `/runs/verify_overlays`, visually inspected (4 of them). Curb traces a
  clean, thin, continuous ribbon exactly along the sidewalk/road boundary on
  real photos, including around curved corners. No sign of mislabeling.
- ✅ **Smoke train on real data** — built a real `vistas8` (`--limit 8`,
  8 train + 8 val real photos, distinct images) and ran the gate. See
  "Smoke train findings" below — passed, but needed one tooling fix first.
- ⬜ A/B (checkpoint × cls_pw) — not started. **Next step.**
- ⬜ Full 100-epoch run — not started.
- ⬜ Final `eval_semantic.py` report — not started (only smoke-tested).

### Bug fixed: `/runs/ultralytics-cfg` didn't pre-exist

First real run showed Ultralytics falling back to `/tmp` for its settings
cache (`user config directory '/runs/ultralytics-cfg/Ultralytics' is not
writable`) even though `compose.yaml` already points `YOLO_CONFIG_DIR` there
and the mount is writable by the container user. Cause: the directory simply
didn't exist yet inside `/runs` and Ultralytics doesn't `mkdir -p` its full
config path before the writability check. Fixed by creating it once
(`mkdir -p /runs/ultralytics-cfg` from inside the container); it now persists
correctly across runs via the bind mount. No code change needed — just a
one-time host-side step, worth knowing if a fresh clone of `/home/megabeast/runs`
ever shows the same warning again.

### Smoke train findings — gate passed, but needed a tooling addition

First attempt (`--epochs 20 --batch 4`, defaults otherwise: `optimizer=auto`,
`cos_lr=True`) came back with `mIoU` stuck flat at ~0.001 across all 20
epochs and `dice_loss` pinned at ~0.998 the entire run — **not** the
"overfit hard on 8 images" the plan's smoke-test gate expected. Root cause,
confirmed by testing, not guessed: with `--limit 8` there are only 2
batches/epoch, so 20 epochs = 40 gradient steps total, and Ultralytics'
`optimizer="auto"` picks `lr0 = round(0.002*5/(4+nc), 6)` — at `nc=124` that's
`7.8e-5`, a rate tuned for the *full* run's thousands of iterations. Running
150 epochs (400 steps) at that same tiny lr did show `mIoU` climbing
monotonically (0.0008 → 0.0119) — proving gradients genuinely flow and
nothing is mechanically broken — just far too slowly to call the gate passed
in a reasonable smoke-test budget.

**Fix:** added `--optimizer` / `--lr0` CLI overrides to `train_semantic.py`
(defaults unchanged — `optimizer` still defaults to `"auto"`, `lr0` still
defaults to `None`/unused, so the full-run config in PLAN.md is untouched).
Re-ran the smoke gate with `--optimizer AdamW --lr0 0.002 --epochs 60`:

- 0 missing masks, 0 corrupt (both train and val scans).
- `Transferred 360/364 items` — deficit exactly 4, as expected.
- No NaN at any point.
- `mIoU`: 0.0008 → 0.068 monotonically; `pixel_acc`: 0.0012 → **0.72**
  monotonically, evaluated on 8 *held-out* real val images (not the 8 train
  images — this is genuine generalization, not just memorization, and it
  happened fast because the ADE20K-pretrained backbone already has strong
  features; only the classifier/aux heads start from fresh init).
- Low headline mIoU (0.068) is expected and *not* a red flag: most of the
  124 classes have zero support at all in an 8-image subset, so their
  per-class IoU is 0 and drags the mean down hard — PLAN.md's evaluation
  section already calls out that the headline mIoU isn't the real
  acceptance criterion, the per-target-class numbers are (`eval_semantic.py`,
  step 6, not yet run — that's for the full run's checkpoint).

**Gate verdict: passed.** Pipeline is mechanically correct end-to-end on
real data. The `--optimizer auto` + tiny-epoch-count combination is smoke-test-
specific weirdness, not a concern for the full 100-epoch run (which has
~1125 batches/epoch at `batch=16` over 18,000 train images — thousands of
iterations, well inside the regime `optimizer="auto"`'s formula is tuned for).

## Exact next steps (copy-paste ready, run from this repo's root)

Verification is done. Next is **PLAN.md step 5, the A/B run** (~1-3h):
10 epochs each of `{cityscapes-init, ade20k-init} × {cls_pw=0, cls_pw=1}`,
compared on per-class IoU for `road`/`sidewalk`/`curb`/`curb cut`.

```bash
# checkpoint x cls_pw grid -- 4 runs, 10 epochs each, full real dataset
docker compose run --rm train train_semantic.py \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --model yolo26s-sem-ade20k.pt --cls-pw 0.0 \
  --epochs 10 --name ab-ade20k-clspw0

docker compose run --rm train train_semantic.py \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --model yolo26s-sem-ade20k.pt --cls-pw 1.0 \
  --epochs 10 --name ab-ade20k-clspw1

docker compose run --rm train train_semantic.py \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --model yolo26s-sem.pt --cls-pw 0.0 \
  --epochs 10 --name ab-cityscapes-clspw0

docker compose run --rm train train_semantic.py \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --model yolo26s-sem.pt --cls-pw 1.0 \
  --epochs 10 --name ab-cityscapes-clspw1

# then eval_semantic.py on each of the 4 checkpoints, compare per-class IoU
# for road/sidewalk/barrier curb/flat curb-cut. Expect: cls_pw=1 may drop
# mean mIoU but should raise curb/curb-cut IoU -- that's the win condition.

# if the A/B picks a winner, THEN the full 100-epoch run (PLAN.md defaults,
# ~6-9h on this GPU):
docker compose run --rm train train_semantic.py \
  --data /datasets/vistas-yolo/vistas-v2.0.yaml \
  --model <winning-checkpoint> --cls-pw <winning-value> \
  --epochs 100 --name vistas124-s-640
```

Note: `--model yolo26s-sem-ade20k.pt` (bare name) will re-download (~13MB,
fast) rather than use the cached copy at `weights/yolo26s-sem-ade20k.pt`,
since that's not on Ultralytics' own resolution path from `/training` cwd.
Harmless — just means the first run of each container does a quick fetch.

## Bugs found and fixed during testing (documented so they aren't rediscovered)

1. **`vistas_convert.py` `--drop-void` palette bug**: the renumbering LUT
   remapped class *indices* but the saved PNG palette still used the
   *original* taxonomy's color ordering — masks would have had visually wrong
   colors (not a training-correctness bug, since Ultralytics only reads the
   raw index, but misleading for anyone eyeballing `--drop-void` output).
   Fixed: palette is now renumbered in lockstep with the LUT.
2. **`vistas_verify.py yaml-check` exception-swallowing bug**: a broken YAML
   (e.g. missing `masks_dir`) crashed with an unhandled `SyntaxError` from
   `check_det_dataset` *before* the accumulated static-check `PROBLEM:`
   messages ever printed — so the most informative diagnostics never showed
   up, just a raw traceback. Fixed: `check_det_dataset` exceptions are now
   caught and folded into the same problems list. Verified with a deliberate
   negative test (missing `masks_dir` + mismatched `nc`) — now prints all
   three problems cleanly and exits 1.
3. **`eval_semantic.py` missing `project`/`name` bug**: without passing them,
   `model.val()` defaults to writing prediction masks/plots under
   `/training/runs/...` — inside the container's own ephemeral filesystem,
   **not** the persistent `/runs` bind mount. Everything would silently
   vanish on container exit. Fixed: `--project`/`--name` args added,
   defaulting to `/runs`/`eval`. Verified the fix lands output in the right
   place.
4. **`train_semantic.py` had no way to override the LR/optimizer** — needed
   to actually exercise the smoke-test gate (see "Smoke train findings"
   above): `optimizer="auto"`'s built-in lr formula is scaled for the full
   run's iteration count and is too conservative to show overfitting on a
   tiny `--limit`-built smoke set in any reasonable epoch budget. Added
   `--optimizer` (default `"auto"`, unchanged) and `--lr0` (default `None`,
   only applied if set) CLI args. Full-run defaults in PLAN.md are untouched.

All four were caught by actually running the code against real and
synthetic fixtures, not just by code review — worth continuing to test each
new step empirically before trusting it, per the pattern so far.

## DONE: datasets/runs moved inside the repo (2026-10-03)

Completed right after the full training run finished (was deferred until
then — see git history for the original TODO). Same filesystem
(`/dev/mapper/ubuntu--vg-ubuntu--lv`), so `mv` was an instant rename, not a
copy, even at 64GB combined:

1. Confirmed no container was using `/datasets` or `/runs` (training had
   already exited cleanly).
2. `mv /home/megabeast/datasets training/datasets` (62G) and
   `mv /home/megabeast/runs training/runs` (2.0G).
3. `docker/compose.yaml`'s two bind-mount defaults now point at
   `../training/datasets` / `../training/runs` (relative to the compose
   file, i.e. inside the repo) instead of the old `/home/megabeast/...`
   paths. Still separately bind-mounted to `/datasets`/`/runs` in the
   container (rather than relying on the whole-repo `/app` mount reaching
   them) so every existing script default and documented path keeps
   working unchanged — override `HACKEYE_DATASETS_DIR`/`HACKEYE_RUNS_DIR` if
   you ever want them elsewhere again.
4. `.gitignore` updated: `training/datasets/` and `training/runs/` added.
5. Re-verified: `vistas_verify.py yaml-check` against the real dataset
   through the new mount (`nc=124`, clean exit), and confirmed all 11
   checkpoints (`epoch0.pt` through `last.pt`) and the full `eval-final/`
   output directory are intact and accessible through the new `/runs`
   mount. `git status` confirms neither new directory is tracked.
