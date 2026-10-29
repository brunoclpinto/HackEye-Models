# Models Lab

A web tool for HackEye datasets. It lets you browse and tag frames, correct masks pixel by pixel (with SAM2
click-to-outline), run bulk class rules, and build taxonomies: create, split, merge and expand classes, then save a merge
map and export it as a new dataset. The current rules (API, data model, safety) are in [CONTRACTS.md](CONTRACTS.md), the history in
[docs/DECISIONS.md](docs/DECISIONS.md), and the approved plans plus the milestone status in [docs/plans/](docs/plans/).
The version is in `lab/VERSION` (v26.1.2, milestone **lab26.1**, developed on branch `lab26.1`).

**Open it at `https://megabeast.tail97a14.ts.net:8443/`** (tailnet only). That address is `tailscale serve --https=8443`
pointing at `127.0.0.1:8766`.

## Running

```bash
cd docker
docker compose up -d lab sam      # lab: CPU, port 127.0.0.1:8766; sam: GPU, internal only, optional
docker compose logs -f lab
docker compose stop sam           # frees the GPU; the lab keeps working, with the SAM tool greyed out
```

- **`lab`** works on the **master dataset** `LAB_DATASET` (default `/datasets/HackEye26.1-124`). Its state lives in
  `<root>/lab/`:
  - `lab.db`: tags, notes, the edit log, recounted class pixels and jobs;
  - `orig/`: the original pseudo-label of each edited frame, saved once before its first change;
  - `thumbs/`.
- **`sam`** runs SAM2.1 large (`~/models/sam2`). It loads on first use and keeps image embeddings for 8 frames, so after
  the first click a click takes about 15 ms. It frees its VRAM after 10 minutes idle (`SAM_IDLE_S`).

## Workflow

**Layout (one page):** Browse sits permanently at the top: filters, plus Prev/Next navigation that wraps around. Below it
is the image, then the sub-groups of the selected menu tab, **Edit | Taxonomy | Rules** (default Edit). Every group
collapses from its header, groups never scroll on their own (the page scrolls), and the image height can be dragged.

**Classes (Edit):** search, then the selected class (shown only once you pick one; click it again to clear). Below,
**Detected** lists the classes in the open frame with their number of detections (separate regions), most first, and
**Undetected** holds the rest by category. The lists update after every edit. Painting tools need a class picked.

1. **Browse (top).** Filter by split, clip, status, confidence, class present or edited. Step with `←`/`→` (or `N`/`P`);
   Next on the last frame goes back to the first, and Prev on the first goes to the last. Tag with `Shift+1` ok, `Shift+2` fix, `Shift+3` drop, `Shift+4` todo, or the buttons.
   `drop` is only a tag; deleting frames is a separate, explicit step.
2. **Edit.** Corrections are saved **into the master masks**: P-mode PNGs written atomically. **Revert** restores the
   original pseudo-label.
3. **Rules.** "Class A → class B" over a filter. Run a dry run first; Apply backs up and logs each frame. The snow → water
   remap is done here.
4. **Taxonomy.**
   - **New class** adds a custom class (ids 124–254) to `<root>/taxonomy.json`. **Split** creates child classes of a class;
     you then assign their pixels with the editor, with "protect source = parent".
   - **Map builder:** load a map (e.g. `HackEye26.1-48`) or start from identity, check classes and **Merge**, **Expand** a
     group, or mark it **void** (`__ignore__`).
   - **Save as new dataset:** the name is **typed by you**, e.g. `HackEye26.1.1-48`; the lab never fills it in. It writes
     `taxonomies/<name>.yaml` and builds `/datasets/<name>/` (hardlinked images, remapped masks, yaml, manifest), with a
     per-class pixel check. **Commit the new map yourself.**
5. **Before training or a Drive copy**, sync the edited frames' class counts and re-finalize:
   `docker compose run --rm -T train datasets/hackeye/sync_lab_edits.py --root /datasets/HackEye26.1-124`

## Editor shortcuts (also behind `?` in the app)

| Keys | Action |
|---|---|
| `V` | Inspect (the default tool): click or tap an item to outline its boundary (holes included) with a label (class · id · px); `Esc` clears, `Enter` relabels it to the active class |
| `B` / `E` / `R` / `L` / `S` | brush / erase to secondary class / region relabel (Shift: whole class) / polygon-lasso / SAM |
| `[` `]`, `Q`, `X` | brush size, round or square, swap active and secondary class |
| wheel, Space-drag or middle-drag, `F`, `0`, `+` `-` | zoom to the cursor (up to 32×), pan, fit, 1:1, zoom |
| `M`, `O`, `G` | mask on/off, outline only, pixel grid (from 8×) |
| `1`–`9`, `/`, Alt+click, Shift+Alt+click | recent classes, search classes, eyedropper, set the protect class |
| SAM: click, right-click, drag, `Tab`, `Enter`, `Esc` | positive point, negative point, box, next candidate, apply, clear |
| `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`, `Ctrl/Cmd+S` | undo, redo, save (moving to another frame autosaves) |
| `←` `→` / `N` `P`, `Shift+1`–`4` | previous or next frame in the filter (wraps around); tag ok, fix, drop, todo |

**Protect source class** applies to every tool: only pixels of that class change. It's the main aid for boundaries, e.g.
painting curb only over sidewalk.

## Tests

```bash
cd docker
docker compose run --rm --entrypoint python lab -m pytest lab/tests datasets/common/tests -q --ignore=lab/tests/e2e
../lab/tests/e2e/run.sh          # Playwright, about 5 min
```

`run.sh` copies 20 real frames to `/runs/lab-e2e/` and runs a throwaway lab on port 8767, so the master never receives
test edits. It cleans up after itself.

## Touch (iPhone or iPad, Chrome or Safari)

- **Two fingers:** pinch to zoom, move to pan.
- **One finger, Inspect tool:** at fit size it scrolls the page; once zoomed in it pans the image. A tap selects an item
  (boundary outline plus label), and a double-tap toggles fit and 4×.
- **Editing tools:** they work with one finger. A second finger cancels the stroke in progress. In SAM, `− point` makes
  the next taps negative.
- **On a phone,** Browse starts collapsed, but Prev/Next and the tags stay in its header.

## Known limits (v26.1.3)

- Touch was tested with Chromium and WebKit emulation (synthetic multi-touch), not yet on a real iPhone.
- On a 900 px high screen the tab's groups start below the fold. Use the recents (`1`–`9`) or collapse or shrink the
  image group.
- Detection counts include single-pixel specks of the pseudo-labels (no minimum size), so busy classes read high.

- A colour chosen when merging is shown only in the builder. An exported class takes the colour of its first (largest)
  member, as in `HackEye26.1-48.yaml`.
- The master yaml has no `nc`, because ids can have gaps after a custom class is deleted. Exports always have dense ids.
- The `sam` image is about 23 GB on disk. It shares the GPU, so stop it during training.
