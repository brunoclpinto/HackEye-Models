# Current status

**Read this first.** See `PLAN.md` for the full design/rationale. This file
tracks what's actually been done and the exact next step, so a fresh session
can resume without re-deriving context.

Last updated: 2026-09-30, after completing all pre-training verification
steps, the real-data smoke-train gate, and discovering + fixing a batch-size
problem that was silently doubling wall-clock. **Next: restart the A/B grid
(PLAN.md step 5) with the now-fixed `batch=8` default.**

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
