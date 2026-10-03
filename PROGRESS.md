# Current status

**Read this first.** See `PLAN.md` for the full design/rationale. This file
tracks what's actually been done and the exact next step, so a fresh session
can resume without re-deriving context.

Last updated: 2026-10-03. **TRAINING COMPLETE.** The full 100-epoch run
(`--model yolo26s-sem-ade20k.pt --cls-pw 0.0 --batch 8 --workers 5
--no-plots --epochs 100 --name vistas124-s-640`) finished cleanly, exit code
0, no crashes across the full ~44h run. Final evaluation run against
`best.pt` — see "FINAL EVALUATION RESULTS" section below for the numbers.
**Next: the deferred dataset/runs-into-repo move (see TODO at bottom) is now
unblocked; otherwise this training phase is done. The curb/curb-cut
fine-tuning pass (deferred, see memory: higher-res-crop-finetuning-for-curb)
is the natural next phase of the project when picked back up.**

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

## TODO: finish moving datasets/runs inside the repo (deferred)

2026-10-03: repo code was reorganized into `docker/`/`training/`/
`datasets/vistas/conversion/`/`models/` (see PLAN.md's Layout section), per
user request. The dataset (`/home/megabeast/datasets`) and training-run
outputs (`/home/megabeast/runs`) were deliberately **left external** for
this pass — user's explicit call: finish the move once the live full
100-epoch run completes, since `/runs` is being actively written to right
now and relocating it mid-write would be destructive.

**Once the run is done, still to do:**
1. Stop/confirm no container is using `/datasets` or `/runs`.
2. Physically move `/home/megabeast/datasets` → `training/datasets/` and
   `/home/megabeast/runs` → `training/runs/` inside this repo (or symlink,
   if the physical data is too large to want inside the repo's own
   filesystem — judge at the time based on available disk space).
3. Update `docker/compose.yaml`'s two bind-mount lines (`HACKEYE_DATASETS_DIR`/
   `HACKEYE_RUNS_DIR` defaults) to point at the new internal paths instead
   of `/home/megabeast/datasets` / `/home/megabeast/runs`.
4. Add `training/datasets/` and `training/runs/` to `.gitignore` (tens of
   GB, must never be tracked).
5. Re-verify with `vistas_verify.py yaml-check` and a quick container start
   that nothing broke.
