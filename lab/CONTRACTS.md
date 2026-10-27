# Models Lab — contracts (v26.1.2, current rules)

The history, the user's decisions and the reasons behind each rule are in [`docs/DECISIONS.md`](docs/DECISIONS.md). The
approved plans are in [`docs/plans/`](docs/plans/).

This file is the single source of truth that `lab/*.py`, `lab/sam/`, `lab/static/`, `datasets/common/` (as the lab
uses it) and `datasets/hackeye/sync_lab_edits.py` are built against. It describes how things work **now**. Where the code
and this text disagree, fix whichever is wrong in the same commit.

**What the Models Lab is:** a web UI in HackEye-Models (reached over Tailscale) for HackEye datasets. With it you browse and
tag frames, correct masks pixel by pixel (with SAM2 click-to-outline), run bulk class rules, and build taxonomies: create,
split, merge and expand classes, save maps, and export them as new datasets.

**Contents:** 1 Data model · 2 Runtime and storage · 3 Database · 4 REST API · 5 SAM service · 6 Frontend ·
7 Safety rules · 8 Testing · 9 Versions

---

## 1. Data model
- **Master dataset:** `LAB_DATASET`, by default `/datasets/HackEye26.1-124`. Masks hold **base ids**: 0–123 are Vistas,
  124–254 are custom classes, 255 is ignore. Edits are written in place.
- **Base taxonomy:** `<root>/taxonomy.json`, with the schema of `taxonomies/vistas124.json`:
  `{id, name, raw_name, hier, color, void, parent}`. It is read and written through `common.taxonomy`.
  - Every change keeps a `.bak` and regenerates the master yaml: names, `label_mapping` voids, path, train and val.
  - The yaml has **no `nc`**, because ids can have gaps after a custom class is deleted.
- **Merge maps:** `taxonomies/*.yaml` are views (merged name → member raw names, fnmatch globs). The reserved group
  `__ignore__` maps its members to 255. Masks never change because of a map, so any merge can be expanded again.
- **Exports:** `/datasets/<name>/`, built by `common.dataset_merge`. Images are hardlinked and masks LUT-remapped, with
  `<name>.yaml` and a manifest. A per-class check requires each merged count to equal the sum of its members' counts, plus
  ignore accounting. The **user types the name**, e.g. `HackEye26.1.1-48`; the lab never fills it in.
- **Syncing to the dataset meta:** `datasets/hackeye/sync_lab_edits.py` writes the DB `classpix` into
  `meta/classpix_<split>.csv`, then re-finalizes. Run it before training or a Drive copy.

## 2. Runtime and storage
- **`lab` compose service** (in `docker/compose.yaml`, built from `lab/Dockerfile` on `python:3.12-slim`):
  - deps from `lab/requirements.txt`: fastapi, uvicorn, numpy, opencv-python-headless, pillow, pyyaml, httpx, pytest;
  - runs as UID:GID with `restart: unless-stopped` on port `127.0.0.1:8766:8766`;
  - mounts the repo at `/app` and `/datasets` read-write, and `/runs`;
  - entrypoint `uvicorn lab.server:app --host 0.0.0.0 --port 8766` with workdir `/app`.
- **Environment:**
  - `LAB_DATASET` (default `/datasets/HackEye26.1-124`): the master root;
  - `LAB_DATASETS_ROOT` (default `/datasets`): where exports go;
  - `LAB_REPO` (default `/app`);
  - `SAM_URL` (default `http://sam:8000`).
- **`sam` compose service** (GPU, no host port):
  - built from `lab/sam/Dockerfile`, which follows `~/media-gen/segmentation/Dockerfile` (CUDA 12.8 runtime, torch cu128,
    sam2 from git);
  - mounts `${MODELS_DIR:-../../../models}/sam2:/models:ro` and `/datasets` read-only;
  - is optional: the lab must work when it is down.
- **Lab state:** `<root>/lab/` holds `lab.db` (SQLite, WAL), `orig/<split>/<stem>.png` (the first-save backups),
  `thumbs/<stem>.jpg` and `.lab-version` (§9).
- **Import path:** `lab/__init__.py` puts `datasets/` on `sys.path`, and the shared code is imported as `common.*`.
- **Tailscale:** `tailscale serve --bg --https=8443 http://127.0.0.1:8766` gives `https://megabeast.tail97a14.ts.net:8443/`.

## 3. Database (`lab/store.py`)
```
frames_state(stem TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'todo',   -- todo|ok|fix|drop|edited
             note TEXT DEFAULT '', edited_at TEXT, edit_count INTEGER DEFAULT 0)
edits(id INTEGER PRIMARY KEY, stem TEXT, at TEXT, tool TEXT, changed_px INTEGER,
      delta TEXT)          -- JSON {"<id>": +/-px} net class change
classpix(stem TEXT PRIMARY KEY, counts TEXT, updated_at TEXT)   -- "id:px;id:px" after the last save (overrides meta csv)
tax_log(id INTEGER PRIMARY KEY, at TEXT, action TEXT, payload TEXT)
jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, progress REAL, started TEXT, finished TEXT,
     params TEXT, result TEXT, log TEXT)   -- kind: export|rule|delete-class; status: running|done|failed
versions(id INTEGER PRIMARY KEY, at TEXT, from_version TEXT, to_version TEXT, wiped TEXT)   -- §9
```
A frame's status becomes `edited` on its first saved edit, unless the user has set `ok`, `fix` or `drop`. The user's tag
wins; `edit_count` still increments.

**Revert** sets the status back to `todo`, resets `edit_count` to 0 and clears `edited_at`. The `edited=1` filter therefore
means `edit_count > 0`. Rule applies and delete-with-remap jobs keep the existing statuses.

## 4. REST API (`lab/server.py`, FastAPI; JSON unless noted; errors carry a `detail` string)
**Meta**
- `GET /api/meta` → `{version, dataset, root, counts:{train,val}, sam:{available, loaded}}`. The SAM health check has a 1 s
  timeout.
- `GET /api/clips` → `[{clip, split, frames}]`.

**Frames.** The source is `frames.csv` (stem, split, clip, k, t_sec, conf_mean, conf_p10, lowconf_frac), with `frames_state`
joined in. Class counts come from the DB `classpix` when present, else `meta/classpix_<split>.csv`. All of it is loaded into
memory at startup.
- `GET /api/frames` with query params:
  - `split`, `clip`, `status`;
  - `cmin`/`cmax` (conf_mean), `lmin`/`lmax` (lowconf_frac);
  - `cls` (base id) with `minpx` (default 2000);
  - `edited` (0/1);
  - `sort`: `time` (clip, k), `conf`, `-conf`, `low` (lowconf_frac, worst first) or `-edited`;
  - `offset`, `limit` (default 200, max 1000).

  Returns `{total, rows:[{stem, split, clip, t_sec, conf_mean, lowconf_frac, status, edit_count}]}`.
- `GET /api/frames/{stem}` takes the same filter params and returns that row plus `{note, prev, next, index, total}`; `prev`
  and `next` are neighbours within the filter.
- `GET /api/frames/{stem}/image.jpg`: the file as-is, `Cache-Control: max-age=86400`.
- `GET /api/frames/{stem}/thumb.jpg`: 320×180 at q80, made on demand and cached.
- `GET /api/frames/{stem}/mask.png`: an **8-bit single-channel PNG of base ids** (not the palette), `Cache-Control: no-cache`.
- `PUT /api/frames/{stem}/mask?tool=<brush|region|polygon|sam|mixed>`: the body is a PNG (`image/png`), 1280×720, grayscale or
  RGB(A) with R=G=B.
  - Rejected with 422 if: the size is wrong, the channels differ, or an id is not in the taxonomy (255 is allowed).
  - On the first change, the current file is copied to `lab/orig/`.
  - It writes a P-mode PNG with the palette from the taxonomy (tmp + os.replace), recounts, updates `classpix`, logs to `edits`
    and updates the state.
  - Returns `{changed_px, counts:{id:px}, edit_count, status}`; when nothing changed it returns `changed_px: 0` and writes
    nothing.
- `POST /api/frames/{stem}/revert`: restores `orig/` if it exists (404 otherwise), recounts, logs `tool=revert`, removes the
  backup and sets the status back to `todo`.
- `PUT /api/frames/{stem}/state` with `{status?, note?}`.

**Taxonomy** (master `taxonomy.json` through `datasets/common/taxonomy.py`)
- `GET /api/taxonomy` → `{classes:[{id, name, raw_name, hier, color, void, parent, custom, frames, pixels}]}`, where `frames`
  counts frames with ≥2000 px.
- `POST /api/taxonomy/classes` with `{name, hier, color:[r,g,b], parent?:id, raw_name?}`:
  - The id is the smallest free one from 124 (409 when full).
  - `raw_name` defaults to `hackeye--<hier>--<slug(name)>`. Names and raw names must be unique (409).
  - It saves the json (with `.bak`), regenerates the master yaml `names`/`nc` (the helper from Phase A) and logs to `tax_log`.
- `POST /api/taxonomy/classes/{id}/split` with `{children:[{name, color}]}`: creates custom classes with `parent=id` and
  `hier=parent.hier`.
- `PATCH /api/taxonomy/classes/{id}` with `{name?, color?, hier?}`: custom classes only (403 for base).
- `DELETE /api/taxonomy/classes/{id}?remap_to=<id>`: custom only. It returns 409 when the class has pixels and no `remap_to`;
  with `remap_to` (255 allowed) it runs a rule job, then deletes. Children of a deleted class inherit its parent. Returns `{deleted}` or `{job}`.

**Maps** (`LAB_REPO/taxonomies/*.yaml`; the format is Phase A's `load_merge_map`/`save_merge_map`, extended by W1 with a
reserved group key `__ignore__` whose members map to 255)
- `GET /api/maps` → `[{name, n_classes}]`.
- `GET /api/maps/{name}` → `{name, groups:{merged_name:[member raw names or globs]}}`.
- `POST /api/maps/preview` with `{groups}` → `{n, classes:[{id, name, color, members:[base ids], void}], lut:[256 ints],
  pixel_share:{merged_id: float}, unassigned:[base ids]}`. Unassigned classes keep their own class. `groups: {}` is the
  identity map (N = number of base classes).
- `POST /api/maps` with `{name, groups, comment?, overwrite?:false}`: the name must match `^[A-Za-z0-9][A-Za-z0-9._-]*$`;
  409 if it exists. Writes `taxonomies/<name>.yaml`.

**Exports**
- `POST /api/exports` with `{name, map, groups?, splits?:["train","val"]}`: 409 if `LAB_DATASETS_ROOT/<name>` exists, or
  if another export is running. When `map` doesn't exist yet, `groups` is saved as that map (404 without `groups`). The job
  runs `common.dataset_merge` in a background thread with 4 worker processes and returns `{job}`. 409 details contain
  "exists".
- `GET /api/jobs/{id}` → the row, with the tail of its log.
- `GET /api/jobs?kind=` → recent jobs.

**Rules**
- `POST /api/rules/dry-run` with `{from:id, to:id, filter:{split?, clip?, stems?}}` → `{frames, pixels, sample:[stems≤20]}`,
  computed from the counts.
- `POST /api/rules/apply` with the same body → `{job}`. It backs up each changed frame, rewrites it, logs `tool=rule:<from>-><to>`
  and keeps the existing statuses.

**SAM proxy**
- `POST /api/sam/embed {stem}` and `POST /api/sam/predict {stem, points:[[x,y]...], labels:[1|0...], box:[x1,y1,x2,y2]|null}`
  are passed to the `sam` service with the frame's `split` added.
- `predict` returns `{masks:[{score, png_b64}]}`. Each `png_b64` is an 8-bit 0/1 PNG of 1280×720, sorted by score; there are
  at most 3.

## 5. SAM service (`lab/sam/app.py`)
- `GET /health` → `{status, device, loaded, cached}`.
- `POST /embed {stem, split}`: loads `/datasets/<dataset>/images/<split>/<stem>.jpg`; the dataset folder name comes from env
  `SAM_DATASET`, default `HackEye26.1-124`. It then runs `set_image` and caches the predictor features keyed by stem in an
  LRU of 8.
- `POST /predict {stem, split, points, labels, box}`: embeds first if needed, then `predict(multimask_output=True)`.
- **Model:** `sam2.1_hiera_large.pt` with `configs/sam2.1/sam2.1_hiera_l.yaml`. It is loaded lazily on first use and unloaded
  (including `torch.cuda.empty_cache()`) after `SAM_IDLE_S` (default 600) seconds idle.
- **Concurrency:** one request at a time (a lock).
- **Image:** `hackeye-models-sam`, about 23 GB on disk. Its Dockerfile sets `USER`/`LOGNAME`/`HOME`, which sam2 needs.
- **Measured (v26.1.0):** the first embed takes about 10.6 s because it loads the model; a warm embed about 260 ms; a
  predict 13–16 ms after a one-off ~5 s warmup. VRAM is about 2 GB loaded and is released on idle unload.

## 6. Frontend (`lab/static/`, vanilla ES modules, no build step)
**Files:**
- `index.html`, `style.css` (copy the token block from `~/repos/HackEye/RnD/lab/static/style.css:1-13` and add the rest);
- `app.js`: the router, the menu tabs **Edit | Taxonomy | Rules** (default Edit), URL hash state `#<tab>/<stem>?<filter>`
  (the frame and filter carry across tabs), and a status bar with the version and SAM state;
- `js/groups.js` (collapsible groups) and `js/nav.js` (the current tab, frame and filter, and hash building);
- `js/api.js` (copy the HackEye Lab's `api.js` as is), `js/util.js` (toast, debounce);
- `js/browser.js`, `js/editor.js`, `js/tools.js`, `js/history.js`, `js/picker.js`, `js/sam.js`, `js/taxonomy.js`,
  `js/rules.js`.

**Page layout (v26.1.1), one page, top to bottom:**
1. A header with the title, the menu tabs Edit | Taxonomy | Rules, and `?`.
2. **Browse group (permanent; not a tab):**
   - The filter bar: split, clip, status, conf range, low range, class with min px, edited, sort, reset, and the count.
   - One navigation row: `← Prev`, the position `i / N` within the filter, `Next →`, and the tag buttons ok, fix, drop and
     todo for the current frame. There are **no thumbnails**.
   - **Wrap-around ("carousel"):** Next on the last frame of the filter goes to the first, and Prev on the first goes to
     the last. When `GET /api/frames/{stem}` returns `next`/`prev` as `null`, the frontend fetches `offset=0` or
     `offset=total-1` with the same filter. A one-frame filter stays put, and an empty filter shows "no frames".
   - The nav row and the tags sit in the group header, so they stay visible when Browse is collapsed; only the filter
     bar folds away. A filter-only hash change (e.g. Reset) refreshes Prev/Next for the new filter.
3. **Image group:** the editor canvas for the current frame, shown on **every** tab.
   - Its height is resizable with a drag handle that's remembered; double-click resets it to the viewport height minus
     400 px.
   - When collapsed, the header still shows the stem, status, edits and clip info.
4. **Work group:** the selected tab's controls, in sub-groups:
   - Edit: Classes · Tools · Protect · View · Frame;
   - Taxonomy: Classes · Map builder · Save;
   - Rules: Rule · Result.
5. The status bar.

**Collapsing:** every group and sub-group, including each hierarchy group in the class picker, collapses from its header.
The state is kept in `localStorage` per group id, inside try/catch. Picker hierarchies with no pixels in the frame start
collapsed.

**Navigation keys (all tabs, when focus is not in an input):** `←`/`→` and `N`/`P` step with wrap-around (autosaving when
dirty), and `Shift+1`–`4` tag the frame ok, fix, drop or todo (matched by `e.code`).

**Edit:**
- **Rendering:**
  - One display canvas, drawn with `setTransform(zoom, 0, 0, zoom, panX, panY)` and `imageSmoothingEnabled=false`.
  - Two offscreen 1280×720 layers: the image, and the mask colourised through a 256-entry RGBA lookup table (as
    `MaskStyle.paint`, `~/repos/HackEye/RnD/lab/static/js/classes.js:84-91`).
  - The grid is a `Uint8Array(1280*720)`, decoded from the mask PNG through a canvas red channel (`overlay.js:236-250`).
  - Updates are dirty-rect only.
- **View controls:**
  - Wheel zooms to the cursor, from fit to 32×. Space-drag or middle-drag pans. `F` fits, `1:1` (`0`) shows actual pixels.
  - A pixel grid appears from 8×.
  - `M` toggles the mask and `O` switches to outline mode, which draws a pixel wherever the class differs from the
    right or bottom neighbour. An opacity slider sits in the toolbar.
- **Readout:** x, y, base id and name under the cursor, plus the merged name when view-through-map is on.
- **Picker (`picker.js`, Edit → Classes):** every class from `/api/taxonomy` as compact chips, grouped by `hier`, each
  group collapsible, with a search box (`/` focuses it and opens the matching groups).
  - Recents 1–9: digit keys always pick from recents.
  - `Alt+click` is the eyedropper.
  - The active class is shown with its swatch.
- **Protect source class:** a dropdown (none, or a class). When it's set, every tool changes only pixels of that class.
  `Shift+Alt+click` sets it from under the cursor.
- **Inspect (`V`, the default tool; the only tool on the Taxonomy and Rules tabs):**
  - A click or tap selects the 4-connected region of the class under the pointer (`Painter.select`, which changes
    nothing).
  - **Box** (HackEye Lab style): a 3 px stroke in the class colour, drawn in screen space with a thin dark outer stroke,
    plus a ~15% tint of the region.
  - **Label tag** filled with the class colour, above the box's top-left corner (inside when there's no room), clamped:
    `<name> · id <id> · <N> px`, plus ` · <merged name>` when view-through-map is on. The text is dark or light by luminance.
  - The selection also shows in the Image header and the status bar.
  - It clears on `Esc`, an edit, undo or redo, or a frame change.
  - `Enter` relabels the selection to the active class, respecting protect, as one undoable patch.
- **Tools (`tools.js`)**, all applied to the grid through `history.record(rect)` beforehand:
  - `B` **Brush:** size 1–64 (`[`/`]`), round or square (`Q`). Strokes interpolate between samples with Bresenham.
    A 1 px square paints exactly one pixel.
  - `E` **Erase to class:** the brush with the "secondary" class (set by right-clicking a picker entry); handy for boundaries.
  - `R` **Region relabel:** click fills the 4-connected component under the cursor (iterative, typed-array stack).
    Shift-click relabels every pixel of that class.
  - `L` **Polygon/lasso:** clicks add vertices and a drag adds freehand points. Enter or double-click closes; the fill is
    scanline even-odd in image pixels. Esc cancels.
  - `S` **SAM:**
    - Left-click and right-click add positive and negative points, and drag makes a box.
    - Each change calls predict, and the best candidate is shown as a hatched highlight; `Tab` cycles the candidates.
    - `Enter` applies the active class (respecting protect) and `Esc` clears.
    - The tool is disabled when SAM is unavailable. Opening a frame prefetches `/api/sam/embed`.
- **Touch (pointer events; iOS Chrome and Safari are WebKit):**
  - **Two fingers, any tool:** pinch zooms around the midpoint (fit to 32×), and moving both pans. WebKit
    `gesturestart`/`gesturechange` are prevented.
  - **One finger, Inspect:**
    - at fit zoom the canvas uses `touch-action: pan-y`, so the page scrolls natively and a tap selects;
    - zoomed in it uses `touch-action: none`, so a drag pans the image and a tap selects.
  - **Double-tap (Inspect only)** toggles fit and 4×.
  - **One finger, editing tools:** the tool works as with a mouse.
    - A second finger rolls back a stroke already started and cancels a lasso or SAM drag.
    - A touch Region fill applies on lift.
    - The touch lasso is freehand drag only.
    - The `− point` button (`sam-negative`) makes the next SAM taps negative.
  - **Phone width (≤ 430 px):**
    - no horizontal scroll; the filter bar wraps; the nav row stays on one line;
    - work sub-groups stack in one column; tap targets are ≥ 36 px;
    - Browse starts collapsed, with the nav row still in its header;
    - the image height fits the width.
- **History (`history.js`):** a per-frame stack of `{x, y, w, h, before, after}` patches, capped at 200. `Ctrl+Z` undoes and
  `Ctrl+Shift+Z` or `Ctrl+Y` redoes.
- **Saving:**
  - `Ctrl+S` saves: the grid is encoded to a PNG (canvas with R=G=B=id, A=255), then `PUT`.
  - Leaving the frame autosaves (on `N`/`P` or navigation) when it's dirty.
  - A "Revert to original" button asks for confirmation.
  - `N`/`P` (or `←`/`→`) move through the filter with wrap-around, and the next two frames' image and mask are preloaded.
- **View-through-map:** a map selector; when it's set, the overlay is coloured by `preview.lut` and editing still works in
  base ids.
- **Status keys:** `Shift+1`–`4` (see Navigation keys). `Cmd+digit` is left to the browser.

**Taxonomy:**
- **Left: the class table** (swatch, id, name, hier, frames, pixel share, custom badge, parent).
  - Buttons: New class, Split (opens a child-name list), Edit and Delete (custom only; delete asks for `remap_to` when the
    class has pixels).
- **Right: the map builder.**
  - Load a map, or start from "identity".
  - Groups appear as cards with their members. Multi-select classes, then Merge (prompts for a name and colour).
  - Expand returns a group's members to identity. A group can be marked void (moved under `__ignore__`).
  - Live preview after each change: N, pixel share and the unassigned classes.
- **Save:**
  - A name input that is **empty by default**, with a placeholder hint `HackEye<yy.major.minor>-<N>`.
  - "Save map" and "Save as new dataset" (export); the export shows job progress and the verify result.

**Taxonomy and the image:** while the Taxonomy tab is open, the live map preview colours the image (view-through-map).
Leaving the tab restores the map chosen under Edit → View.

**Rules:** from, to, filter, then Dry run, which shows counts and sample thumbnails (clicking one opens it in the image);
Apply shows job progress.

**Additional keys:** `G` toggles the pixel grid, `X` swaps the active and secondary class, `+`/`-` zoom, and `?` shows every shortcut.
**Test hooks:** the `data-testid` attributes listed in `lab/tests/e2e/conftest.py` (v26.1.1 adds `nav-prev`,
`nav-next`, `nav-pos` with `data-index` and `data-total`, `tag-ok|fix|drop|todo`, `group-<id>` with `data-collapsed`, and
`group-toggle`; v26.1.2 adds `tool-inspect`, `sam-negative` and `selection-label`, `selection` in `state()`
(`{id, px, bbox:[x0,y0,x1,y1] inclusive, label}` or null) and `clientToImg`; `tab-browse`, `carousel` and `frame-card` are gone), and `window.__lab` with `state()`,
`imgToClient(x,y)` and `getPixel(x,y)`. Keep them stable, because the E2E suite depends on them.
**Merge colour:** the colour chosen in the map builder only shows on the card swatch. An exported merged class takes its
first member's colour, because the map format has no colour field.

## 7. Safety rules
- **Mask writes:** P-mode PNG, written atomically (tmp + os.replace) with the taxonomy palette. They are validated as
  1280×720 with every id in the taxonomy or 255. An upload whose channels differ is rejected.
- **Backups:** each frame's original is backed up once, to `orig/`, before its first change. Revert restores the backup and
  deletes it.
- **Locking:** one process-wide lock covers every write to masks, the taxonomy and the DB.
- **Frame deletion:** `drop` is only a tag; deleting frames is an explicit step outside the UI (an `apply_curation.py`-style
  keep list).
- **Custom classes:** one with pixels can only be deleted with a remap target.
- **Tests:** they never edit the master. E2E runs on a fixture copy in `/runs/lab-e2e/`.

## 8. Testing
**Unit (`lab/tests/`, pytest + `fastapi.testclient`):**
- A `conftest.py` fixture (with `labhelpers.py`) generates a tiny synthetic master in `tmp_path`: 4 train and 2 val frames at 1280×720, random
  blocks of classes, frames.csv, meta csvs, taxonomy.json (vistas124 plus nothing custom) and yaml. Point `LAB_DATASET` and
  `LAB_REPO` at it, with `taxonomies/` holding a small test map.
- Cover: every endpoint's happy path and its 4xx cases, mask round-trip and palette, backup once and revert, recount and
  filters, class create/split/patch/delete guard, map preview and save, an export job end to end on the fixture (hardlinks,
  count check), and rule dry-run vs apply.
- Run: `cd docker && docker compose run --rm --entrypoint python lab -m pytest lab/tests -q`.

**E2E (`lab/tests/e2e/`, Playwright, adapted from `~/repos/HackEye/RnD/tests/e2e/run.sh`):**
- `make_fixture.py` copies 20 real frames from the master (including 3 with "nature snow", both splits) into
  `/runs/lab-e2e/HackEye-fixture`.
- `run.sh` starts a throwaway lab (`docker compose run -d --rm -p 127.0.0.1:8767:8766 -e LAB_DATASET=... -e
  SAM_DATASET=...`) and runs the tests against `LAB_URL=http://127.0.0.1:8767`. Afterwards it stops the throwaway lab and
  deletes the fixture.
- Tests:
  - browse, filter and tag;
  - open the editor, zoom ≥8×, brush 1 px, undo, redo, then save; check the PNG on disk and the backup through the API
    (mask.png bytes);
  - region relabel; polygon; protect mode leaves other classes untouched;
  - SAM click, then apply (skipped when SAM is unavailable);
  - revert;
  - create a class and paint with it; split a class;
  - load HackEye26.1-48, merge two groups, expand one, preview N, then save as a new dataset into the fixture's datasets root
    and wait for the job: the export yaml exists and the count check passes;
  - a rules dry run of snow → water returns frames ≥ 3.

## 9. Versions and version-scoped content (`lab/VERSION`, `lab/versioning.py`)
- Versions are `v<YY>.<major>.<minor>`, with the minor dropped when 0 in prose. `lab/VERSION` holds the current one; it is
  `v26.1.2` for milestone **lab26.1** (mask correction, SAM2, taxonomy builder).
- Each milestone has its own branch (`lab26.1`). After it is squash-merged into `main`, the branch is deleted locally and
  remotely.
- **At startup** `check_version` compares `lab/VERSION` with `<root>/lab/.lab-version`:
  - when the file is missing, it is written and nothing is wiped;
  - on a change, **only regenerable lab state** is wiped: `<root>/lab/thumbs/` and the `jobs` rows. A `versions` row
    records it.
- **Never wiped:** masks, `orig/` backups, `frames_state` (tags, notes), `edits`, `classpix`, `tax_log`, `taxonomy.json`
  and the yaml. These are the user's curation of the dataset, unlike the HackEye Lab's runs and reviews.
- Jobs still `running` at startup are marked `failed` with "interrupted by restart".
