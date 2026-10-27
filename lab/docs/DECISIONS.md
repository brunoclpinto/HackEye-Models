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
