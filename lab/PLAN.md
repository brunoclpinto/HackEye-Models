# Models Lab: plan and contract (v0.1.0)

Status: **v0.1.0 built (2026-10-27).** Phase A (the taxonomy restructure, commit 7b32817) and Phase B (this contract) are done.
35 unit tests and 10 E2E tests pass. Usage is in README.md. Deviations from the contract: the master yaml has no `nc`;
`POST /api/exports` takes an optional `groups`; merge colours are not persisted; the identity map is `groups: {}`.
The approved design is copied in "Design" below. Everything after it is the contract the workers build to. The workers
are Sonnet; Opus reviews and integrates.

## Design (approved 2026-10-27)
The Models Lab is a web tool in this repo for HackEye datasets. With it the user:
- browses and tags frames;
- corrects masks with pixel accuracy (brush, region relabel, polygon/lasso, SAM2 click-to-outline, protect-source-class);
- runs bulk class rules;
- builds taxonomies: creates classes, splits a class into children, merges classes into a map, expands (ungroups) a merge,
  and saves the map plus an exported dataset under a name the user types (e.g. `HackEye26.1.1-48`; the lab never auto-names).

The user works on a Mac with a mouse over Tailscale. Every one of the 124 classes and any custom class can be picked.

**Master dataset.** It is `/datasets/HackEye26.1-124`, and edits happen there.
- Masks hold base ids: 0–123 are Vistas, 124–254 are custom classes, 255 is ignore.
- `<root>/taxonomy.json` defines the base classes; Phase A's `datasets/common/taxonomy.py` reads and writes it.
- Merge maps live in `taxonomies/*.yaml` and are views; masks never change because of a merge.
- An export builds `/datasets/<name>/` (hardlinked images, LUT-remapped masks) through `datasets/common/dataset_merge.py`.

## Runtime
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
- **Lab state:** `<root>/lab/` holds `lab.db` (SQLite, WAL), `orig/<split>/<stem>.png` (the first-save backups) and
  `thumbs/<stem>.jpg`.
- **Tailscale (done by Opus):** `tailscale serve --bg --https=8443 http://127.0.0.1:8766`.

## Database (`lab/store.py`)
```
frames_state(stem TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'todo',   -- todo|ok|fix|drop|edited
             note TEXT DEFAULT '', edited_at TEXT, edit_count INTEGER DEFAULT 0)
edits(id INTEGER PRIMARY KEY, stem TEXT, at TEXT, tool TEXT, changed_px INTEGER,
      delta TEXT)          -- JSON {"<id>": +/-px} net class change
classpix(stem TEXT PRIMARY KEY, counts TEXT, updated_at TEXT)   -- "id:px;id:px" after the last save (overrides meta csv)
tax_log(id INTEGER PRIMARY KEY, at TEXT, action TEXT, payload TEXT)
jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, progress REAL, started TEXT, finished TEXT,
     params TEXT, result TEXT, log TEXT)   -- kind: export|rule|delete-class; status: running|done|failed
```
A frame's status becomes `edited` on its first saved edit, unless the user has set `ok`, `fix` or `drop`. The user's tag
wins; `edit_count` still increments.

## API (JSON unless noted; errors are `HTTPException` with a `detail` string)
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
  - `sort`: `time` (clip, k), `conf`, `-conf`, `low` or `-edited`;
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
  - The id is the next free one from 124 (409 when full).
  - `raw_name` defaults to `hackeye--<hier>--<slug(name)>`. Names and raw names must be unique (409).
  - It saves the json (with `.bak`), regenerates the master yaml `names`/`nc` (the helper from Phase A) and logs to `tax_log`.
- `POST /api/taxonomy/classes/{id}/split` with `{children:[{name, color}]}`: creates custom classes with `parent=id` and
  `hier=parent.hier`.
- `PATCH /api/taxonomy/classes/{id}` with `{name?, color?, hier?}`: custom classes only (403 for base).
- `DELETE /api/taxonomy/classes/{id}?remap_to=<id>`: custom only. It returns 409 when the class has pixels and no `remap_to`;
  with `remap_to` it runs a rule job, then deletes. Returns `{deleted}` or `{job}`.

**Maps** (`LAB_REPO/taxonomies/*.yaml`; the format is Phase A's `load_merge_map`/`save_merge_map`, extended by W1 with a
reserved group key `__ignore__` whose members map to 255)
- `GET /api/maps` → `[{name, n_classes}]`.
- `GET /api/maps/{name}` → `{name, groups:{merged_name:[member raw names or globs]}}`.
- `POST /api/maps/preview` with `{groups}` → `{n, classes:[{id, name, color, members:[base ids], void}], lut:[256 ints],
  pixel_share:{merged_id: float}, unassigned:[base ids]}`. Unassigned classes keep their own class (the current convention).
- `POST /api/maps` with `{name, groups, comment?, overwrite?:false}`: the name must match `^[A-Za-z0-9][A-Za-z0-9._-]*$`;
  409 if it exists. Writes `taxonomies/<name>.yaml`.

**Exports**
- `POST /api/exports` with `{name, map, splits?:["train","val"]}`: 409 if `LAB_DATASETS_ROOT/<name>` exists. Otherwise it saves
  the map if it's new, then starts the job (`dataset_merge` in a thread or subprocess) and returns `{job}`.
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

## SAM service (`lab/sam/app.py`)
- `GET /health` → `{status, device, loaded, cached}`.
- `POST /embed {stem, split}`: loads `/datasets/<dataset>/images/<split>/<stem>.jpg`; the dataset folder name comes from env
  `SAM_DATASET`, default `HackEye26.1-124`. It then runs `set_image` and caches the predictor features keyed by stem in an
  LRU of 8.
- `POST /predict {stem, split, points, labels, box}`: embeds first if needed, then `predict(multimask_output=True)`.
- **Model:** `sam2.1_hiera_large.pt` with `configs/sam2.1/sam2.1_hiera_l.yaml`. It is loaded lazily on first use and unloaded
  (including `torch.cuda.empty_cache()`) after `SAM_IDLE_S` (default 600) seconds idle.
- **Concurrency:** one request at a time (a lock).

## Frontend (`lab/static/`, vanilla ES modules, no build step)
**Files:**
- `index.html`, `style.css` (copy the token block from `~/repos/HackEye/RnD/lab/static/style.css:1-13` and add the rest);
- `app.js`: the router and tabs **Browse | Edit | Taxonomy | Rules**, URL hash state such as `#edit/<stem>?<filter>`, and a
  status bar with the version and SAM state;
- `js/api.js` (copy the HackEye Lab's `api.js` as is), `js/util.js` (toast, debounce);
- `js/browser.js`, `js/editor.js`, `js/tools.js`, `js/history.js`, `js/picker.js`, `js/sam.js`, `js/taxonomy.js`,
  `js/rules.js`.

**Browse:**
- Filter bar (split, clip, status, conf range, low range, class, edited, sort).
- A virtual-scrolled grid of thumbnails with the stem, conf and a status badge.
- Click selects; Enter or double-click opens the editor. Keys `1 ok`, `2 fix`, `3 drop`, `4 todo` tag the selection and
  arrows move it.

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
- **Picker (`picker.js`):** every class from `/api/taxonomy`, grouped by `hier`, with a search box (`/` focuses it).
  - Recents 1–9: digit keys pick from recents in the editor (they tag only in the Browse tab).
  - `Alt+click` is the eyedropper.
  - The active class is shown with its swatch.
- **Protect source class:** a dropdown (none, or a class). When it's set, every tool changes only pixels of that class.
  `Shift+Alt+click` sets it from under the cursor.
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
- **History (`history.js`):** a per-frame stack of `{x, y, w, h, before, after}` patches, capped at 200. `Ctrl+Z` undoes and
  `Ctrl+Shift+Z` or `Ctrl+Y` redoes.
- **Saving:**
  - `Ctrl+S` saves: the grid is encoded to a PNG (canvas with R=G=B=id, A=255), then `PUT`.
  - Leaving the frame autosaves (on `N`/`P` or navigation) when it's dirty.
  - A "Revert to original" button asks for confirmation.
  - `N`/`P` move through the Browse filter, and the next two frames' image and mask are preloaded.
- **View-through-map:** a map selector; when it's set, the overlay is coloured by `preview.lut` and editing still works in
  base ids.
- **Status keys:** `Ctrl+1..4` set ok, fix, drop and todo.

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

**Rules:** from, to, filter, then Dry run, which shows counts and sample thumbnails; Apply shows job progress.

## Tests
**Unit (`lab/tests/`, pytest + `fastapi.testclient`):**
- A `conftest.py` fixture generates a tiny synthetic master in `tmp_path`: 4 train and 2 val frames at 1280×720, random
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

## Worker split (Phase B, disjoint files)
- **W1, backend:**
  - `lab/server.py`, `store.py`, `masks.py`, `rules.py`, `taxonomy_api.py`, `export.py`, `lab/tests/` (unit) and
    `lab/requirements.txt`;
  - `datasets/hackeye/sync_lab_edits.py`, which writes DB `classpix` rows into `meta/classpix_<split>.csv`, then runs
    finalize;
  - the `__ignore__` extension in `datasets/common/taxonomy.py`.
- **W2, frontend:** all of `lab/static/`.
- **W3:** `lab/sam/` (Dockerfile, app.py), `lab/Dockerfile`, the `docker/compose.yaml` services `lab` and `sam`, and
  `lab/tests/e2e/` (run.sh, conftest, make_fixture.py, tests).
- **Every worker:** don't commit; no stray containers or files; report the files changed and the test output.
