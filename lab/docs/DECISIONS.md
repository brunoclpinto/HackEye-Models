# Models Lab — decision log

Each version gets one entry: what prompted it, the decisions made with the user, the implementation choices and deviations
the workers made, known limitations, and the measured result. The current rules are in
[`../CONTRACTS.md`](../CONTRACTS.md); this file explains **why** they are what they are. The approved plans are in
[`plans/`](plans/).

**How we work** (user decisions, standing; they follow the HackEye Lab's rules):
- **Versioning:** `v<YY>.<major>.<minor>`, with the minor dropped in prose when it is 0. `lab/VERSION` holds the current
  version.
- **Milestones:** each milestone has its own branch (`lab26.1`). After it is squash-merged into `main`, the branch is deleted
  locally and remotely.
- **Version-scoped content:** a version change wipes only regenerable lab state (thumbnails, job history). The user's
  curation (mask edits, backups, tags, notes, edit log, taxonomy) is never wiped. This differs on purpose from the HackEye
  Lab, whose runs and reviews are throwaway (CONTRACTS §9).
- **Roles:** Opus plans, writes the contracts, reviews, integrates and verifies; Sonnet workers implement, and every plan
  names them. Delivery for "commit, push, mail" is:
  1. tests (unit and E2E);
  2. DECISIONS and CONTRACTS;
  3. bump `lab/VERSION`;
  4. restart the lab (only when no job is running);
  5. commit on the milestone branch;
  6. push;
  7. mail the user a summary.
- **Don't block on results:** misses go under *Known / possible issues*. Only destructive steps wait for the user.
- **Questions by mail:** a question mail includes a recommended default, and work continues with it. The Gmail connector is
  send-only for now, so the user answers in the terminal.
- **Naming:** dataset exports are named by the user (e.g. `HackEye26.1.1-48`); the lab never auto-names.

---

## v26.1 — Models Lab: mask editor, SAM2 assist, taxonomy builder (commits `7b32817`, `d1ebfb6`, plus the v26.1.0 docs commit)
**Prompt.** The pseudo-labelled `HackEye26.1-124` (17,347 frames after curation) needs correcting before retraining. Known
errors: whole regions flipped (surf, foam and glare labelled "snow"; rare surfaces swallowed by road or sidewalk) and rough
boundaries. CVAT was judged too heavy and poor for dense 124-class masks. The HackEye Lab belongs to the app repo's version
cycle. So the user asked for a separate lab in HackEye-Models, with pixel-level boundary correction. Later the same day they
added: create new classes, merge and expand them, and save as a new dataset yaml, "like the 124->48 i did earlier".

**Decisions with the user (2026-10-27):**
- **Device:** a Mac with a mouse or trackpad, so the design is mouse-first with keyboard shortcuts.
- **Classes:** the picker lists all 124 Vistas classes plus custom ones; masks keep base ids.
- **SAM2** is in from v26.1.0, not deferred.
- **One tool:** browsing and tagging are in the lab (no FiftyOne).
- **"Expand"** means both undoing a merge and splitting a class into new sub-classes.
- **Naming:** the master keeps its name, and exports get a version-style name that the user decides by hand
  (e.g. `HackEye26.1.1-<N>`).
- **Branch and docs:** the lab commits were moved off `main` to `lab26.1`. `main` was reset to `6a17416` and
  force-pushed with lease. The HackEye Lab's documentation and version rules were adopted.

**Design choices (Opus):**
- **Merge maps are non-destructive views.** Masks always keep base ids, so any merge can be expanded again. Exports are
  derived datasets (hardlinked images, LUT-remapped masks), as `vistas_merge.py` did for the 48-class set.
- **Repo restructure:** `taxonomies/` holds `vistas124.json` and `HackEye26.1-48.yaml`, and `datasets/common/` holds
  `taxonomy.py` and `dataset_merge.py`. The proof is that the 124→48 LUT stays byte-identical: sha256 `a84b7dd1…`, recorded
  from the old `build_merge_lut` before the move and checked against the master's `taxonomy.json` after it.
- **Edits are written into the master masks**, so training uses them directly. A one-time backup per frame goes to
  `<root>/lab/orig/`, and the edit log goes to SQLite. `drop` is only a tag.
- **SAM2 runs as its own GPU service**, reusing the media-gen recipe and the `~/models/sam2` weights. It caches embeddings per
  frame and frees VRAM when idle, so the lab keeps working without it.
- **Version wipe adapted:** only thumbnails and job history are wiped (see "How we work").

**Worker deviations and contract notes:**
- **Phase A:** unlisted void classes keep their own new id and stay ignored through `label_mapping`; they don't map to 255,
  which is what keeps the LUT identical. `Taxonomy` allows id gaps. `dataset_merge` finds the source yaml by
  `--src-yaml`, then `<src>.yaml`, then the only `*.yaml` in the root.
- **W1 (backend):**
  - The master yaml has no `nc`, because of id gaps.
  - `POST /api/exports` takes an optional `groups`.
  - The export accounting counts pixels of `__ignore__` classes, which required an edit to `dataset_merge.py`.
  - Sort `low` is worst first.
  - Revert resets `edit_count`.
  - New class ids take the smallest free id from 124.
- **W2 (frontend):**
  - The merge colour isn't persisted (the map format has no colour field).
  - Identity is `groups: {}`, and loaded globs are expanded to explicit members.
  - Extras beyond the plan: `G` grid, `X` swap, a per-frame note box.
  - The first E2E failures were root-relative static paths, fixed to `/static/...`.
  - Loading a map no longer creates a duplicate `__ignore__` group from `void--unlabeled`; that bug turned N from 48 into 47.
- **W3 (infra and E2E):** the SAM image needed `USER`/`LOGNAME`/`HOME` for sam2. Tests use `data-testid` attributes and the
  `window.__lab` hooks (listed in `lab/tests/e2e/conftest.py`).
- **Integration fixes (Opus, from screenshots):**
  - The identity-map preview was rejected (`groups` had to be non-empty), which raised an error toast on the Taxonomy tab.
  - The editor's info bar printed "null".
- **Version mechanics:** `lab/VERSION`, `lab/versioning.py`, and a `versions` table. Jobs still running at startup are marked
  failed.

**Result:**
- **Tests:** 39 unit tests (lab plus datasets/common) and 10 Playwright E2E tests on a 20-frame fixture copy all pass. The
  master received no test edits; `orig/` is absent and the edit count is 0.
- **SAM timings:** the first embed takes 10.6 s (it loads the model), a warm embed 260 ms, and a predict 13–16 ms. VRAM is
  about 2 GB while loaded and drops to about 0.2 GB after the idle unload.
- **Access:** served at `https://megabeast.tail97a14.ts.net:8443/`.

**Known / possible issues:**
- The merge colour isn't persisted; an exported class takes its first member's colour.
- The `sam` image is about 23 GB on disk. It shares the GPU with training, so stop it during training runs.
- The HackEye repo's `common/README.md` (branch `v26.2`) points at `taxonomies/`, which exists only on `lab26.1` until
  the merge.
- Untested so far: long real editing sessions, SAM quality on thin structures such as curbs and poles, and performance with
  many custom classes.

## v26.1.1 — single-page layout: permanent browse with carousel, image, work panel (branch `lab26.1`)
**Prompt.** The user asked (2026-10-27): "browse is not a menu option, its permanent with filters at top, carroussel
navigation. below we have the image. below image we have the selected menu option work stuff. menu order is correct and
defaults to edit. each group should be collapsable."

**Decisions with the user:**
- Browse is no longer a tab. It's a permanent group at the top: filters plus a carousel.
- Below it is the image, then the selected tab's work panel. The menu is Edit | Taxonomy | Rules, and Edit is the default.
- Every group is collapsible.
- **Defaults proposed by mail and agreed by the user in the terminal:**
  - the image stays visible on all tabs;
  - tagging moves to `Shift+1`–`4`, since digits pick recent classes and `Cmd+digit` belongs to the Mac browser;
  - the class picker moves below the image as compact chips grouped by hierarchy.

**Worker choices (Sonnet):**
- The hash is `#<tab>/<stem>?<filter>`, so the frame and filter carry across tabs.
- The live Taxonomy preview colours the image only while that tab is open.
- The image height defaults to the viewport height minus 400 px; the drag handle is remembered and a double-click resets it.
- Picker hierarchies with no pixels in the frame start collapsed, and searching opens the matches.
- New files `js/groups.js` and `js/nav.js`.
- E2E:
  - `test_browse.py` was rewritten (filter and tag, carousel navigation, collapse that persists after a reload);
  - helpers `pick_class` and `to_client` were added;
  - no result assertion was weakened.

**Opus review (screenshots at 1440×900 and 1920×1080):** the layout matches the request. Carousel cards cut off the start of
the stem, so every card read "Chest-center…". They now show `C-0001 · 000215`, with the full stem in the tooltip.

**Result:** 39 unit tests and 12 E2E tests pass (the SAM test ran). The API is unchanged.

**Known / possible issues:**
- At 900 px high, the Work panel starts below the fold, so picking a class chip means scrolling. Recents (`1`–`9`), a
  collapsed or shorter image group, or a later "sticky image" option reduce this.
- `Cmd+digit` can't be used for tagging on the Mac, so tagging is on `Shift+1`–`4`.

**Follow-up: navigation (same version, v26.1.1, at the user's request).** The user clarified: "navigation of images is next
and previous, no carroussel thumbs. when i said carroussel i meant infinite navigation, so last next goes to first and
first previous goes to last. don't bump version for this fix."
- The thumbnail carousel, the first/last buttons and `Home`/`End` were removed.
- The Browse group is now the filters plus one row: Prev, `i / N`, Next, and the tags.
- Navigation wraps in the frontend: when `next`/`prev` is null, it fetches the first or last frame of the filter. The API
  is unchanged.
- Plan: [`plans/v26.1.1-followup-navigation.md`](plans/v26.1.1-followup-navigation.md).
- The E2E tests check the wrap-around in both directions.

## v26.1.2 — touch support and the click-to-inspect box (branch `lab26.1`)
**Prompt.** The user tested in Chrome on iOS: "image area does not accept pinch zoom nor scroll, it should. when i click an
item in the image need bounding box with classe name like its used in hackeye lab." They also set a standing rule: "go
for it" now means the full drill (plan, execute, tests, docs, version, commit, push, mail).

**Decisions (Opus, under "go for it"):**
- **Inspect (`V`) is a new tool and the default**, so a click or tap never paints by accident. An "item" is the
  4-connected region of the class under the pointer. The box and label copy the HackEye Lab's `drawHazard` look, and the
  label reads `<name> · id · px` (plus the merged name when view-through-map is on). `Enter` relabels the selection, which
  makes Inspect a quick way to fix a whole wrong region.
- **Touch:**
  - two fingers pinch and pan with any tool;
  - one finger in Inspect scrolls the page at fit (`pan-y`) and pans when zoomed;
  - editing tools take one finger, and a second finger rolls back the stroke;
  - SAM gets a negative-point toggle;
  - the layout works at phone width.
- **Version bump to v26.1.2:** this is new capability, unlike the v26.1.1 navigation fix.

**Worker choices (Sonnet):**
- Browse starts collapsed on phones (its nav row stays in the header).
- Double-tap works only in Inspect.
- A touch Region fill applies on lift, so a second finger can still cancel it.
- New `__lab` hooks: `selection` and `clientToImg`.
- New E2E files `test_inspect.py` and `test_touch.py`; the touch tests run on **both Chromium touch emulation and WebKit**.

**Opus review (screenshots):**
- Desktop 1440×900: a sidewalk selection showed a magenta box with "flat sidewalk · id 24 · 139759 px".
- iPhone 12 Pro Max emulation (428×926 @3×): zoomed in, with a car selected ("vehicle car · id 108 · 3942 px").
- Both looked correct.

**Result:** 39 unit tests and 28 E2E tests pass: inspect (checked against a flood fill of the fixture mask), Esc, Enter with
undo, redo, save and protect, pinch keeping the midpoint within 1 px, two-finger pan, one-finger pan, tap, double-tap,
stroke rollback, the SAM toggle, and no horizontal scroll at 428 px.

**Known / possible issues:**
- **Not verified on a real iPhone:**
  - whether iOS hands the first finger to a page scroll at fit zoom;
  - pinches whose first finger drifts vertically;
  - real `gesturestart` suppression;
  - double-tap timing;
  - the iOS toolbar and safe areas.
- The tests synthesise multi-touch with PointerEvents rather than real hardware.

**Follow-up: outline (same version, v26.1.2, at the user's request).** "i don't want binding box, i need boundary lines like
in hackeye lab."
- The rectangle is gone. The selection is drawn by contours traced along pixel edges, the outer boundary plus holes,
  in the HackEye Lab's outline style: a dark under-stroke, a 3 px class-colour stroke and a ~15% tint. The label sits at the
  top-left of the contour extent.
- `selection` gains `contours` and `edgeLen`.
- New E2E checks:
  - `edgeLen` equals the 4-neighbour boundary edge count computed from the mask;
  - a region with a hole has at least 2 contours;
  - canvas sampling shows the stroke on the boundary and only the tint inside.
- **Screenshots:**
  - the sidewalk on `Chest-center-0001_000000`: 3 contours, the manholes outlined as holes;
  - a 10× close-up: the line follows the pixel steps;
  - the iPhone emulation with a car selected.
- **Possible tuning after the user's real-phone test:** on small items at phone size, the dark under-stroke can look heavier
  than the class colour on dark classes such as car blue.
- Plan: [`plans/v26.1.2-followup-outline.md`](plans/v26.1.2-followup-outline.md). Tests: 39 unit, 29 E2E.

---

## v26.1.3 — no Work group, full-height groups, Detected / Undetected classes with counts, selected class only when picked
**Prompt (2026-10-29, "go for it").** "drop work group, the subgroups are enough; groups and subgroups are never
scrollable inline, just occupy the height they need to occupy; classes divided into 2 parallel subgroups detected (classes
found in selected image) and undetected (classes not found in selected image); classes should have the amount of
detections in front of name like the class merge group; love the selected color and class next to search bar but should
only be there if a class is actually selected and only a selected class."

**Decisions (Opus, from the request):**
- **Work group dropped:** the tab panes sit directly on the page; the tab bar already says which tab is open.
- **No inline scrolling:** the class list, the taxonomy table and the map builder lost their `max-height`; the page is the
  only vertical scroller. Sideways scrolling of the taxonomy table on phones stays (it isn't vertical, and the table is
  wider than a phone).
- **"Detection" = item:** a 4-connected region of one class, the same thing Inspect outlines, with no minimum size, so the
  number matches what a click shows. The tooltip adds the pixel total.
- **Detected is flat and sorted by count** (the busiest classes first, which is where corrections start); **Undetected**
  keeps the hier groups, collapsed, because it holds ~100 classes.
- **"Only a selected class":** no class is active on load (before, the first class was silently active, which is why the
  head always showed one). The "Erase to" box and the recents, which also sat in the head, moved to Tools. Painting with
  no class changes nothing and toasts once. Clicking the active chip again clears it.

**Implementation (Sonnet worker):** new `js/counts.js` (one iterative flood-fill pass over the grid); `picker.js`
rewritten to build the chips once and move them between the lists; `editor.js` recounts on frame open, after every
committed edit, undo and redo (a plain save changes no pixels, so it doesn't recount), and guards the painting tools.
Worker choices: the hier-group counts under Undetected count only their undetected members; a lasso without a class toasts
when it is closed, not per vertex; `X` with no active class takes the secondary.

**Opus review (screenshots, live lab):** desktop 1440 px on `Chest-center-0001_000000`: Detected 22 (barrier curb 11,
flat sidewalk 9, flat road 8, ...) beside Undetected 102 in 8 hier groups, with "nature snow (62)" next to the search box
after picking it; iPhone 390 px: the lists stack, no horizontal scroll. Labelling took 9.3 ms.

**Result:** 28 unit tests and 35 E2E tests pass (6 new in `test_classes.py`: active class hidden / shown / cleared;
Detected equals the classes in the fixture mask on disk; chip counts equal `cv2.connectedComponents(connectivity=4)`;
painting a new class moves it to Detected with count 1 and undo moves it back; no class = no change + toast; no inline
vertical scroller on Edit and Taxonomy; no Work group). The earlier entries' "39 unit tests" was a miscount: the suite
has had 28 tests since v26.1.0.

**Known / possible issues:**
- Counts include the pseudo-labels' single-pixel specks, so noisy classes read high (e.g. "flat road 8" on a frame with
  one road). A minimum size could be added if that's distracting.
- With the class list at full height, the Tools / View / Frame groups on a phone sit below ~100 undetected chips unless
  their hier groups stay collapsed (they are by default).
- Not yet seen on a real iPhone (same as v26.1.2).
- Plan: [`plans/v26.1.3.md`](plans/v26.1.3.md).

---

## v26.1.4 — scroll-view pan, clear the class, selected-class-only mask, numbered detections
**Prompt (2026-10-29, "go for it").** "i can zoom the image but not drag it into position, it should behave like a mobile
phone scrollview where i can zoom and drag image around to focus on specific locations; should be able to unselect
selected class; when no class is selected show all masks otherwise just the selected class mask; below selected class
should show numbered list of detections for fine-selection." Follow-up before shipping: "keep zoom on mouse wheel, move
image with ctrl+mouse down".

**Cause (Opus):** with a mouse or trackpad an Inspect press selected at once and a drag did nothing; panning needed
Space-drag or a middle-drag, which nothing on screen hinted at. Touch already panned when zoomed.

**Decisions:**
- **Pan:** Ctrl/Cmd + left drag in any tool (the user's choice), plus a plain drag in Inspect when zoomed, selecting on
  release instead of on press. The wheel keeps zooming (the first plan had a plain scroll pan; the user reverted that).
  Touch pans glide after lift (fling), like a phone scroll view. Double-click toggles fit and 4× as double-tap does.
- **Clear the class:** an `×` on the selected-class box, plus page `Esc` after any pending lasso / SAM / selection.
- **Mask:** a selected class shows only its own pixels, so its detections stand out; no class shows all.
- **Detections:** numbered by size, largest first (the numbers stay stable while viewing, and the big ones come first);
  the same numbers are tagged on the image so the list and the picture match. Clicking one selects and zooms to it
  (bbox ≈ half the view, fit..8×). More than 60 fold behind `+N more` instead of scrolling (no inline scroll, v26.1.3).

**Implementation (Sonnet worker):** `counts.js` `classRegions()` adds per-region `{k, px, bbox, seed}` to the same single
pass; `editor.js` unifies mouse, pen and touch Inspect on the tap/drag path, adds the Ctrl-pan, fling, class filter in
`paintRect`, tags and `selectDetection`; `picker.js` adds `×` and the Detections row. Worker choices: mouse double-click
reuses the double-tap timing (320 ms, 30 px) instead of `dblclick`, so it never fires twice; selecting a detection also
scrolls the image into view (helps on phones where the chips are below it).

**Opus review (screenshots, live lab, read-only):** desktop, `flat sidewalk` on `Chest-center-0001_000000`: only the
sidewalk mask, tags 1–9 on the image, chips `#1 139759 px` … `#9 1 px`, `×` next to the name. Clicking `#2` (298 px)
zoomed to 8× and outlined it beside the parked car. iPhone 390 px: the Detections row wraps under the class.

**Opus review fix:** a pinch moves the view through zoom, and the first fling measured that as pan velocity, so lifting a
pinch could fling the image to the pan limit (it showed as an intermittent failure of the two-finger pan test). The
two-finger fling now follows the finger midpoint and is skipped when the zoom changed in the last 80 ms; new test
`test_pinch_leaves_no_fling`.

**Result:** 28 unit tests and 46 E2E tests pass, three full runs in a row (new `test_view.py`: Inspect drag pans and doesn't select, click selects
on release, the wheel zooms with the cursor point fixed, Ctrl-drag with Brush pans and paints nothing, double-click
fit/4×, touch fling glides then stops and a new press stops it; `test_classes.py`: `×` and the `Esc` order, the
selected-class-only mask via `maskPixel`, detections equal `cv2.connectedComponentsWithStats(connectivity=4)` sorted by
size and `#1` selects the largest inside the view, the absent class and `+N more`).

**Known / possible issues:**
- On macOS, Ctrl+click is a secondary click; the canvas suppresses the context menu, but a system-wide setting that
  remaps Ctrl+click could still interfere. Cmd+drag does the same as a fallback.
- A tag can sit under the Inspect label of a selected neighbour (the label is drawn on top).
- Plan: [`plans/v26.1.4.md`](plans/v26.1.4.md).

---

## v26.1.5 — view colour for contrast, label size
**Prompt (2026-10-29, "go for it").** After asking how to change a class colour (only custom classes can be edited in
the Taxonomy tab): "the class colour can stay fixed but i need to change it while editing for contrast purposes allow me
to do so in view group. in view group add a label size slidder, i need bigger labels sometimes as well."

**Decisions (Opus):**
- **Display-only override, per viewer:** the colour lives in the browser (`localStorage`), never in `taxonomy.json`, so
  the class colour stays fixed for maps, exports and the app, as the user asked. It survives reloads because contrast
  needs are usually per class (a magenta sidewalk on pink paving), not per frame.
- **Target:** the selected class, else the class of the Inspect selection, so it works in both workflows.
- **Everything the editor draws** for that class follows it (mask, outlines, labels, tags, swatches), so the chips still
  match the image.
- **Label size:** 10–32 px for the Inspect label; detection tags scale at 0.85× so they stay smaller than the label.

**Implementation (Sonnet worker):** `store.js` holds the overrides (`viewColor`, `setViewColor`, a `viewcolors` event);
`editor.js` routes every drawing colour through it and rebuilds the LUT on change; `picker.js` swatches follow it. Opus
grouped the colour controls so `Reset` doesn't wrap away from the picker.

**Opus review (screenshot, live lab, read-only):** `flat sidewalk` set to #00ff66 with labels at 26 px: detection #2
zoomed to 8× with a large green label, green tags and green chip swatches; the Taxonomy colour unchanged.

**Test fix:** `test_one_finger_drag_pans_when_zoomed_and_tap_selects` failed now and then since v26.1.4, because the
fling could move the view between the lift and the check. The test now rests the finger 120 ms before lifting (no fling
by design), which also covers that rule.

**Result:** 28 unit tests and 49 E2E tests pass, five full runs in a row (new `test_view_style.py`: the mask pixel and swatch change while
`/api/taxonomy` doesn't, reload keeps it, Reset restores it, the target follows the Inspect selection and the control is
disabled with neither, the label grows from 13 to 26 px and persists).

**Known / possible issues:**
- The override is per browser: the iPhone and the Mac keep separate colours.
- Plan: [`plans/v26.1.5.md`](plans/v26.1.5.md).

---

## v26.1.6 — Filters, a Frame group with a side column, hover descriptions
**Prompt (2026-10-30, "go for it").** "add quick description to buttons when hovering; browse group rename to filters.
move the buttons on the right (that match frame group below image) and add it to image group right side and add a save
button to it (save will save current changes to file); remove frame group; rename image group to frame; move tools
protect and view to the left of the new frame group, max occupancy 25%; from tools remove the erase to and history +
current selection history bit."

**Decisions (Opus):**
- **Group ids kept** (`browse`, `image`) so stored collapse states and test ids survive the renames.
- **Frame header:** the tags and Prev/Next moved from the Filters header, plus Save (the existing save: master mask file,
  originals kept in `lab/orig`). The old Frame sub-group went entirely.
- **Removed with the Frame sub-group:** the note field, Undo/Redo and Revert buttons. Undo/Redo stay on the keyboard; Revert
  stays in the API. Mailed the user whether any should come back (default: no).
- **Side column only on Edit**, since Tools/Protect/View were edit-only; ≤ 600 px stacks it above the image.
- **"Erase to" and recents removed from the UI only:** right-click still sets the secondary class (chip style), `X` swaps,
  `1`–`9` pick recents.

**Implementation (Sonnet worker):** `index.html` regrouped; `#eSide` toggled per tab in `app.js`; `editor.js` dropped the
removed wiring and now highlights the current status on `#bTags`; `picker.js` dropped its extra host; titles added in
`index.html`, `groups.js` ("Collapse or expand <name>"), `taxonomy.js` and `util.js` dialogs. **Deviation:** three CSS rules
(a reserved line for the view-colour hint, a full-width colour row, a stable selection label row on phones) so the image
doesn't move when an item is selected; without them the touch double-tap test failed at phone width.

**Opus review:** screenshots at 1400 px (Edit, Taxonomy) and 390 px (Edit) match the plan. Fix: button blur after a click
now covers the side column as well as the Edit pane (keyboard shortcuts would otherwise re-press a focused button).

**Result:** 39 unit tests (the 28 in earlier notes counted only `lab/tests`) and 55 E2E tests pass (new `test_layout.py`, 6 tests:
titles and header contents, Save clean/dirty/written, side column ≤ 25% and left of the canvas, hidden on Taxonomy,
phone stacking, every visible button titled).

**Known / possible issues:**
- Notes and Revert are not reachable from the page until the user decides.
- The side column at 25% of a narrow laptop window wraps its controls onto more lines.
- Plan: [`plans/v26.1.6.md`](plans/v26.1.6.md).
