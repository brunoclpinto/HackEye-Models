"""Touch: pinch zoom around the midpoint, two-finger pan, one-finger pan when zoomed, tap select, double-tap, phone layout.
Runs on Chromium touch emulation and on WebKit (iOS engine). Pinches use dispatched PointerEvents (pointerType touch)."""
import math

from conftest import BASE, find_pixel, mask_array, open_editor, pick_class, state, tid, to_client

PTR = """(a) => { const c = document.querySelector('[data-testid="editor-canvas"]');
  c.dispatchEvent(new PointerEvent(a.type, {pointerId: a.id, pointerType: 'touch', isPrimary: a.id === 1, button: 0,
    clientX: a.x, clientY: a.y, bubbles: true, cancelable: true})); }"""


def fire(page, type_, id_, x, y):
    page.evaluate(PTR, {"type": type_, "id": id_, "x": x, "y": y})


def open_touch(page, stem):
    open_editor(page, stem)
    page.evaluate("document.querySelector('[data-testid=\"editor-canvas\"]').scrollIntoView({block: 'nearest'})")


def canvas_mid(page):
    b = page.locator(tid("editor-canvas")).bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def pinch(page, mx, my, d0, d1, steps=12):
    fire(page, "pointerdown", 1, mx - d0, my); fire(page, "pointerdown", 2, mx + d0, my)
    for i in range(1, steps + 1):
        d = d0 + (d1 - d0) * i / steps
        fire(page, "pointermove", 1, mx - d, my); fire(page, "pointermove", 2, mx + d, my)
    fire(page, "pointerup", 1, mx - d1, my); fire(page, "pointerup", 2, mx + d1, my)


def under(page, cx, cy):
    return page.evaluate("([x, y]) => window.__lab.clientToImg(x, y)", [cx, cy])


def screen_of(page, fx, fy):
    return page.evaluate("([x, y]) => window.__lab.imgToClient(x - 0.5, y - 0.5)", [fx, fy])      # imgToClient is a pixel centre


def test_no_horizontal_scroll_and_viewport_meta(touch_page, frames):
    open_touch(touch_page, frames[0]["stem"])
    assert touch_page.evaluate("document.querySelector('meta[name=viewport]').content") == "width=device-width, initial-scale=1"
    w = touch_page.evaluate("[document.documentElement.scrollWidth, document.body.scrollWidth, document.querySelector('#page').scrollWidth, window.innerWidth]")
    assert max(w[:3]) <= w[3], w
    for tab in ("taxonomy", "rules"):
        touch_page.goto(f"{BASE}/#{tab}")
        touch_page.wait_for_timeout(600)
        w = touch_page.evaluate("[document.documentElement.scrollWidth, document.querySelector('#page').scrollWidth, window.innerWidth]")
        assert max(w[:2]) <= w[2], (tab, w)
    # tap targets of the main controls are at least 36 px tall
    touch_page.goto(f"{BASE}/#edit")
    touch_page.wait_for_selector(tid("nav-next"))
    for t in ("nav-prev", "nav-next", "tag-ok", "tool-inspect"):
        assert touch_page.locator(tid(t)).bounding_box()["height"] >= 35.5, t
    assert not touch_page.errors


def test_pinch_zoom_keeps_midpoint_and_two_finger_pan(touch_page, frames):
    open_touch(touch_page, frames[1]["stem"])
    mx, my = canvas_mid(touch_page)
    z0 = state(touch_page)["zoom"]
    p0 = under(touch_page, mx, my)
    pinch(touch_page, mx, my, 20, 100)
    z1 = state(touch_page)["zoom"]
    assert z1 > z0 * 3, (z0, z1)                               # 5x finger spread
    s = screen_of(touch_page, p0["fx"], p0["fy"])
    assert abs(s["x"] - mx) <= 1 and abs(s["y"] - my) <= 1, (s, mx, my)
    # pinching back in returns toward fit and never below it
    pinch(touch_page, mx, my, 100, 5)
    assert state(touch_page)["zoom"] >= z0 - 1e-6
    # two-finger pan: both fingers move together by (+40, +20): the image follows
    pinch(touch_page, mx, my, 20, 80)
    a = screen_of(touch_page, 640, 360)
    fire(touch_page, "pointerdown", 1, mx - 30, my); fire(touch_page, "pointerdown", 2, mx + 30, my)
    for i in range(1, 9):
        fire(touch_page, "pointermove", 1, mx - 30 + 5 * i, my + 2.5 * i); fire(touch_page, "pointermove", 2, mx + 30 + 5 * i, my + 2.5 * i)
    fire(touch_page, "pointerup", 1, mx + 10, my + 20); fire(touch_page, "pointerup", 2, mx + 70, my + 20)
    b = screen_of(touch_page, 640, 360)
    assert abs((b["x"] - a["x"]) - 40) <= 1.5 and abs((b["y"] - a["y"]) - 20) <= 1.5, (a, b)
    assert not touch_page.errors


def test_one_finger_drag_pans_when_zoomed_and_tap_selects(touch_page, frames):
    open_touch(touch_page, frames[2]["stem"])
    mx, my = canvas_mid(touch_page)
    pinch(touch_page, mx, my, 20, 90)
    assert state(touch_page)["zoom"] > 1
    a = screen_of(touch_page, 640, 360)
    fire(touch_page, "pointerdown", 3, mx, my)
    for i in range(1, 7):
        fire(touch_page, "pointermove", 3, mx + 8 * i, my + 4 * i)
    touch_page.wait_for_timeout(120)                           # the finger rests before lifting: no fling (v26.1.4)
    fire(touch_page, "pointerup", 3, mx + 48, my + 24)
    touch_page.wait_for_timeout(100)
    b = screen_of(touch_page, 640, 360)
    assert abs((b["x"] - a["x"]) - 48) <= 1.5 and abs((b["y"] - a["y"]) - 24) <= 1.5, (a, b)
    assert state(touch_page)["selection"] is None              # a drag is not a tap
    # a real tap (touch events) selects the item under the finger
    cx, cy = canvas_mid(touch_page)
    touch_page.touchscreen.tap(cx, cy)
    touch_page.wait_for_function("() => window.__lab.state().selection !== null")
    px = under(touch_page, cx, cy)
    assert state(touch_page)["selection"]["id"] == touch_page.evaluate("([x, y]) => window.__lab.getPixel(x, y)", [px["x"], px["y"]])
    assert not touch_page.errors


def test_double_tap_toggles_fit_and_4x(touch_page, frames):
    open_touch(touch_page, frames[3]["stem"])
    mx, my = canvas_mid(touch_page)
    fit = state(touch_page)["zoom"]
    for _ in range(2):
        fire(touch_page, "pointerdown", 1, mx, my); fire(touch_page, "pointerup", 1, mx, my)
    assert abs(state(touch_page)["zoom"] - 4) < 0.01
    for _ in range(2):
        fire(touch_page, "pointerdown", 1, mx, my); fire(touch_page, "pointerup", 1, mx, my)
    assert abs(state(touch_page)["zoom"] - fit) < 0.01


def test_second_finger_rolls_back_a_started_stroke(touch_page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=0)
    open_touch(touch_page, stem)
    pick_class(touch_page, 0)
    touch_page.keyboard.press("b")
    touch_page.select_option(tid("brush-shape"), "square")
    touch_page.fill(tid("brush-size"), "64")
    c = to_client(touch_page, x, y)
    fire(touch_page, "pointerdown", 1, c["x"], c["y"])
    fire(touch_page, "pointermove", 1, c["x"] + 10, c["y"] + 5)
    assert touch_page.evaluate("([x, y]) => window.__lab.getPixel(x, y)", [x, y]) != old     # the stroke painted
    fire(touch_page, "pointerdown", 2, c["x"] + 60, c["y"])                                  # second finger: gesture, stroke rolled back
    fire(touch_page, "pointerup", 2, c["x"] + 60, c["y"]); fire(touch_page, "pointerup", 1, c["x"] + 10, c["y"] + 5)
    assert touch_page.evaluate("([x, y]) => window.__lab.getPixel(x, y)", [x, y]) == old
    assert not state(touch_page)["dirty"]


def test_sam_negative_toggle(touch_page, frames):
    open_touch(touch_page, frames[0]["stem"])
    b = touch_page.locator(tid("sam-negative"))
    assert "on" not in (b.get_attribute("class") or "")
    b.click()
    assert "on" in (b.get_attribute("class") or "")
    b.click()
    assert "on" not in (b.get_attribute("class") or "")
