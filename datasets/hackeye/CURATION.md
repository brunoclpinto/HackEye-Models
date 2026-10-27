# HackEye frames dataset: curation (near-duplicate trimming)

Status: **dry run done (2026-10-27); waiting for the user to pick a config.** Nothing is deleted or moved until then.

## Results (full train set, 169,696 frames)

| config | kept | % kept | protected extras | hard-review | mean conf kept |
|---|---|---|---|---|---|
| fps2_sim0.75 | 8,016 | 4.7% | 1,942 | 233 | 0.869 |
| fps2_sim0.85 | 12,570 | 7.4% | 935 | 92 | 0.872 |
| fps2_sim0.90 | 14,830 | 8.7% | 406 | 39 | 0.874 |
| fps2_sim0.95 | 16,472 | 9.7% | 67 | 6 | 0.876 |
| fps4_sim0.75 | 12,981 | 7.6% | 3,370 | 937 | 0.858 |
| fps4_sim0.85 | 22,330 | 13.2% | 2,218 | 534 | 0.863 |
| fps4_sim0.90 | 27,272 | 16.1% | 1,127 | 230 | 0.864 |
| fps4_sim0.95 | 32,073 | 18.9% | 223 | 34 | 0.867 |

The mean conf over all frames is 0.853. The full report, with per-clip and per-class tables, is
`~/media/hackeye-models/datasets/HackEye/curation/report.md`.

Findings:
- "Within 25%" (sim 0.75) is too loose. Large classes (sky, road, building) dominate the grid, so different
  scenes merge, and protection then has to add many frames back.
- Trimming keeps the class mix almost unchanged. It removes redundancy but does not fix class imbalance.
- Picking by confidence raises the mean conf only a little (0.853 to 0.864–0.876).

Rerun: `cd docker && docker compose run --rm -T --entrypoint python -e CUDA_VISIBLE_DEVICES= train datasets/hackeye/curate.py --root /datasets/HackEye --split train`
(about 1 min the first time, a few seconds with the grid cache).

## Why

The frames are 20 fps, so neighbouring frames are near-identical. Standing still (at a crossing, at a door) gives
hundreds of copies of one scene. These overweight that scene in training and waste epochs. The user asked
(2026-10-26) to trim frames with similar detections and keep the most accurate frame of each similar run.

## Method (`curate.py`, CPU only, dry run)

1. **Fixed subsample (stage A).** Per clip, split the 20 fps index `k` into bins of `20 / base_fps` frames. Keep the
   frame with the highest `conf_mean` in each bin. Swept values: `base_fps` 2 and 4.
2. **Layout grid.** Each mask (1280×720) is reduced to a 32×18 grid of 40×40 blocks. Each cell holds the block's
   majority class after the HackEye26.1-48 merge, so subtype flips (sign types, marking variants) don't count as
   differences. Grids are cached, so a re-run only computes new masks.
3. **Layout dedup (stage B).** Walk the stage-A frames of each clip in time order. A frame joins the current group
   when its grid agrees with the group's first frame (the anchor) on at least `sim` of the 576 cells, and it is no
   more than `max_gap` s (default 10) after the anchor. Otherwise it starts a new group. Groups never cross clips.
   Each group keeps its most confident frame: highest `conf_mean`, ties broken by lower `lowconf_frac`. "Most
   accurate" can only mean "most confident", since there is no ground truth. Swept values: `sim` 0.75 (the user's
   "within 25%"), 0.85, 0.90 and 0.95.
4. **Rare-class protection.** A frame "has" a class when that class covers at least `protect_px` pixels in it
   (default 2000), counted from `classpix`. Protected classes are:
   - a named navigation list (Vistas ids): curb 4, bike-lane 13, crosswalk-plain 14, curb-cut 15,
     pedestrian-area 19, rail-track 20, road-shoulder 22, service-lane 23, traffic-island 25, crosswalk-zebra 45
     and 56, stop-line 51, bicycle symbol 52, catch-basin 69, manhole 74, pothole 77, pedestrian traffic light 91;
   - any class present in fewer than 1% of the split's frames.

   If a group has a protected class that its kept frame lacks, the group also keeps the best frame for that class.
   This preserves class coverage without keeping every frame.
5. **Hard frames, for review.** Confidence-based picking favours easy frames. So per group, the least confident
   dropped frame goes to `hard_review.csv` when its `lowconf_frac` is at least 0.15. These frames are not in the
   kept set; they are candidates for human review.
6. **Val is not trimmed.** Its frames line up with the Lab test timestamps. The report only gives its counts.

## Outputs

These go in `~/media/hackeye-models/datasets/HackEye/curation/`, and nothing in the dataset is changed:
- `grid_cache_<split>.npz`: the cached grids;
- `<config>/keep_train.csv` (stem, group, reason), `<config>/groups.csv`, `<config>/hard_review.csv`;
- `report.json` and `report.md`: the sweep table with, per config:
  - frames kept and % kept;
  - kept per clip;
  - extra frames kept by protection;
  - the hard-review count;
  - the mean `conf_mean` of kept frames vs all frames;
  - per merged class, the share of frames that have it and the pixel share, before vs after.

## Worker recommendation

Sonnet. This is one self-contained numpy/PIL script with a precise spec. Opus reviews the diff and the numbers.
