"""Fixtures for the Models Lab browser tests (lab/CONTRACTS.md §8).

Run through run.sh: a throwaway lab on a 20-frame fixture (3 frames with nature snow, id 62; both splits).

HOOKS AND data-testid ATTRIBUTES THE FRONTEND MUST PROVIDE (W2). Tests use these and nothing else (plus the contract's keys:
B E R L S tools, Q shape, Ctrl+Z / Ctrl+Shift+Z undo/redo, Ctrl+S save, Enter/Esc, Shift+1..4 tag, N/P and Left/Right next/prev).

Global hook (editor page), `window.__lab`:
  state()            -> {stem, zoom, tool, cls, dirty, samCandidates}   zoom = display zoom factor, tool = 'brush'|'erase'|
                        'region'|'polygon'|'sam', cls = active base id, dirty = unsaved edits, samCandidates = number of SAM masks shown
  imgToClient(x, y)  -> {x, y}  client (viewport) coordinates of the CENTRE of image pixel (x, y)
  clientToImg(cx, cy)-> {x, y, fx, fy} image pixel (floored) and float image coords under a client point
  getPixel(x, y)     -> base id of the editor's mask grid at that pixel
  state().selection  -> null or {id, px, bbox:[x0,y0,x1,y1] inclusive, label}  (Inspect selection)
Layout (v26.1.1, one page: Browse group, Image group, Work group; tabs Edit | Taxonomy | Rules, default Edit; no Browse tab):
                    tab-edit tab-taxonomy tab-rules      sam-status (data-available="true|false")
                    group-<id> (every group/sub-group; attr data-collapsed="true|false"; ids browse image work edit-classes edit-tools
                    edit-protect edit-view edit-frame tax-classes tax-map tax-save rules-rule rules-result hier-<hier>)
                    group-toggle (the header button inside a group; the group's own is the first one)
Browse (always visible):
                    filter-split filter-clip filter-status filter-class (all <select>, option value = filter value; for the class
                    filter the value is the base id)   frame-count (text = total of the current filter)
                    nav-prev nav-next (buttons; wrap: Next on the last frame -> first, Prev on the first -> last)
                    nav-pos (text "i / N", attrs data-index = 1-based position of the current frame in the filter, data-total = N)
                    tag-ok tag-fix tag-drop tag-todo (buttons: tag the current frame; Shift+1..4 too)
                    Keys: Left/Right (or P/N) step with wrap-around, Shift+1..4 tag
Editor:             editor-canvas (the display canvas)   editor (root; data-dirty="true|false")
                    tool-inspect (default tool) sam-negative (toggle: next SAM taps are negative)  selection-label (text of the box label)
                    tool-brush tool-erase tool-region tool-polygon tool-sam (buttons)
                    brush-size (<input>)   brush-shape (<select> values round|square)
                    picker-search (<input>)  picker-item (one per class; attr data-id = base id)  active-class (data-id)
                    protect-select (<select>, option value = base id or "" for none)
                    btn-save  btn-revert (asks window.confirm)  readout (text under the cursor)
Taxonomy:           class-row (attrs data-id data-custom)   btn-new-class btn-split btn-edit-class btn-delete-class
                    dlg-name (first text input of any dialog)  dlg-hier  dlg-color (<input type=color> or text '#rrggbb')
                    dlg-child-name (one input per child; split dialog starts with 2)  dlg-ok
                    map-load (<select>, value = map name)   group-card (attr data-name)  group-select (checkbox inside a card)
                    btn-merge (merges the selected group cards)  group-expand (button inside a card)  preview-n (text = N)
                    map-name (<input>)  btn-save-map  btn-export  job-status (attr data-status = running|done|failed, text has verify)
Rules:              rule-from rule-to (<select>, option value = base id)  btn-dry-run
                    rule-result (attrs data-frames data-pixels)
Native dialogs (confirm/prompt/alert) are accepted automatically by the page fixture; a prompt is answered with the default.
"""
import io
import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = os.environ.get("LAB_URL", "http://127.0.0.1:8767")
DATA = os.environ.get("E2E_DATA", "/e2e-data")      # fixture root's parent (read-only): HackEye-fixture/, datasets/
SNOW = 62
W, H = 1280, 720


def api(path, method="GET", body=None, raw=False, data=None, ctype=None):
    payload = json.dumps(body).encode() if body is not None else data
    headers = {"Content-Type": "application/json"} if body is not None else ({"Content-Type": ctype} if ctype else {})
    req = urllib.request.Request(BASE + path, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            b = r.read()
            return b if raw else (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        raise AssertionError(f"{method} {path} -> {e.code} {e.read()[:300]!r}") from None


def mask_array(stem):
    """The frame's mask as seen through the API: HxW uint8 of base ids."""
    return np.array(Image.open(io.BytesIO(api(f"/api/frames/{stem}/mask.png", raw=True))))


def put_mask(stem, arr, tool="brush"):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return api(f"/api/frames/{stem}/mask?tool={tool}", "PUT", data=buf.getvalue(), ctype="image/png")


@pytest.fixture(scope="session", autouse=True)
def _lab():
    meta = api("/api/meta")
    assert meta["counts"]["train"] + meta["counts"]["val"] == 20, meta
    return meta


@pytest.fixture(scope="session")
def sam_available(_lab):
    return bool(api("/api/meta")["sam"]["available"])


@pytest.fixture(scope="session")
def frames(_lab):
    rows = api("/api/frames?sort=time&limit=1000")["rows"]
    assert len(rows) == 20
    return rows


@pytest.fixture(scope="session")
def snow_stems(_lab):
    rows = api(f"/api/frames?cls={SNOW}&minpx=2000&limit=1000")["rows"]
    assert len(rows) >= 3, "fixture must hold 3 frames with snow"
    return [r["stem"] for r in rows]


@pytest.fixture(scope="session")
def classes(_lab):
    return {c["raw_name"]: c for c in api("/api/taxonomy")["classes"]}


@pytest.fixture
def clean_frame(frames):
    """Hand out untouched frames one at a time (each test gets a different one; edits are reverted afterwards)."""
    used = []

    def take(prefer=None):
        pool = [prefer] if prefer else [r["stem"] for r in frames]
        for s in pool:
            if s not in used:
                used.append(s)
                return s
        raise AssertionError("no free frame")
    yield take
    for s in used:
        try:
            api(f"/api/frames/{s}/revert", "POST")
        except AssertionError:
            pass            # 404: never edited
        api(f"/api/frames/{s}/state", "PUT", {"status": "todo"})


@pytest.fixture(scope="session")
def pw():
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(pw):
    b = pw.chromium.launch(headless=True)
    yield b
    b.close()


@pytest.fixture(scope="session", params=["chromium", "webkit"])
def touch_browser(pw, request):
    """Touch tests run on Chromium touch emulation and on WebKit (what iOS Chrome/Safari use), when WebKit can launch."""
    try:
        b = getattr(pw, request.param).launch(headless=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{request.param} not available: {e}")
    yield b
    b.close()


@pytest.fixture
def touch_page(touch_browser):
    """iPhone 12 Pro Max-sized touch context (428x926, hasTouch, isMobile)."""
    ctx = touch_browser.new_context(viewport={"width": 428, "height": 926}, device_scale_factor=2, has_touch=True, is_mobile=True)
    pg = ctx.new_page()
    pg.set_default_timeout(30_000)
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("dialog", lambda d: d.accept())
    yield pg
    ctx.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1500, "height": 950})
    pg = ctx.new_page()
    pg.set_default_timeout(30_000)
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("dialog", lambda d: d.accept())
    yield pg
    ctx.close()


# ---- page helpers --------------------------------------------------------

def tid(name):
    return f'[data-testid="{name}"]'


def lab(page, expr, *args):
    return page.evaluate(f"(a) => window.__lab.{expr}", list(args))


def state(page):
    return page.evaluate("window.__lab.state()")


def open_editor(page, stem):
    page.goto(f"{BASE}/#edit/{stem}")
    page.wait_for_selector(tid("editor-canvas"))
    page.wait_for_function(f"() => window.__lab && window.__lab.state().stem === '{stem}'")
    page.wait_for_function("() => window.__lab.getPixel(0, 0) !== undefined")   # mask grid decoded


def to_client(page, x, y):
    # the page scrolls (picking a class chip below the image scrolls it away): bring the canvas back first
    page.evaluate("document.querySelector('[data-testid=\"editor-canvas\"]').scrollIntoView({block: 'nearest'})")
    return page.evaluate("([x, y]) => window.__lab.imgToClient(x, y)", [x, y])


def pixel(page, x, y):
    return page.evaluate("([x, y]) => window.__lab.getPixel(x, y)", [x, y])


def pick_class(page, cid):
    # chips of hierarchy groups without pixels in the frame start collapsed; searching by id expands the matching group
    page.fill(tid("picker-search"), str(cid))
    page.locator(f'{tid("picker-item")}[data-id="{cid}"]').first.click()
    page.fill(tid("picker-search"), "")
    page.evaluate("document.activeElement.blur()")        # keys (B R L S ...) must not go into the search box
    assert state(page)["cls"] == cid


def zoom_to(page, x, y, minimum=8.0):
    """Wheel-zoom on image pixel (x, y) until the zoom is >= minimum (zoom follows the cursor, so the pixel stays under it)."""
    for _ in range(40):
        if state(page)["zoom"] >= minimum:
            return
        c = to_client(page, x, y)
        page.mouse.move(c["x"], c["y"])
        page.mouse.wheel(0, -300)
        page.wait_for_timeout(60)
    raise AssertionError(f"zoom stayed at {state(page)['zoom']}")


def click_pixel(page, x, y, button="left"):
    c = to_client(page, x, y)
    page.mouse.click(c["x"], c["y"], button=button)


def drag_pixels(page, pts):
    c = [to_client(page, x, y) for x, y in pts]
    page.mouse.move(c[0]["x"], c[0]["y"])
    page.mouse.down()
    for p in c[1:]:
        page.mouse.move(p["x"], p["y"], steps=6)
    page.mouse.up()


def find_pixel(arr, not_class=None, is_class=None, margin=40):
    """A pixel (x, y) away from the borders whose 9x9 neighbourhood is a single class, not `not_class` / equal to `is_class`."""
    for y in range(margin, H - margin, 8):
        for x in range(margin, W - margin, 8):
            v = arr[y - 4:y + 5, x - 4:x + 5]
            c = int(v[4, 4])
            if (v == c).all() and (is_class is None or c == is_class) and (not_class is None or c != not_class):
                return x, y, c
    return None
