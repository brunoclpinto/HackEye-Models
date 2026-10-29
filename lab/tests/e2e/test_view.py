"""v26.1.4: scroll-view navigation. Mouse drag pans the zoomed image (Inspect), a click selects on release, a plain wheel pans
(Ctrl/Cmd+wheel zooms, at fit the page scrolls), double-click toggles fit/4x, and a touch pan keeps gliding after lift."""
from conftest import find_pixel, mask_array, open_editor, pick_class, state, tid, zoom_to
from test_touch import canvas_mid, fire, open_touch, pinch, screen_of, under


def centre(page):
    page.evaluate("document.querySelector('[data-testid=\"editor-canvas\"]').scrollIntoView({block: 'nearest'})")
    b = page.locator(tid("editor-canvas")).bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def zoomed(page, stem):
    arr = mask_array(stem)
    x, y, _ = find_pixel(arr)
    open_editor(page, stem)
    zoom_to(page, x, y, 4.0)
    return x, y


def test_mouse_drag_pans_zoomed_image_and_selects_nothing(page, frames):
    zoomed(page, frames[0]["stem"])
    cx, cy = centre(page)
    z0 = state(page)["zoom"]
    p0 = page.evaluate("([x, y]) => window.__lab.clientToImg(x, y)", [cx, cy])
    page.mouse.move(cx, cy); page.mouse.down()
    page.mouse.move(cx - 70, cy - 35, steps=8)
    page.mouse.up()
    st = state(page)
    assert st["zoom"] == z0 and st["selection"] is None
    s = page.evaluate("([x, y]) => window.__lab.imgToClient(x - 0.5, y - 0.5)", [p0["fx"], p0["fy"]])     # imgToClient is a pixel centre
    assert abs(s["x"] - (cx - 70)) <= 1 and abs(s["y"] - (cy - 35)) <= 1, (s, cx, cy)
    # a click without moving selects, on release
    page.mouse.move(cx, cy); page.mouse.down()
    assert state(page)["selection"] is None
    page.mouse.up()
    assert state(page)["selection"] is not None
    # a tiny wobble (< 6 px) is still a click
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + 3, cy + 2); page.mouse.up()
    assert state(page)["selection"] is not None
    assert not page.errors


def test_wheel_zooms_at_cursor_and_ctrl_drag_pans_with_brush(page, clean_frame):
    stem = clean_frame()
    arr = mask_array(stem)
    x, y, _ = find_pixel(arr)
    open_editor(page, stem)
    cx, cy = centre(page)
    page.mouse.move(cx, cy)
    p0 = page.evaluate("([x, y]) => window.__lab.clientToImg(x, y)", [cx, cy])
    z0 = state(page)["zoom"]
    page.mouse.wheel(0, -300)
    assert state(page)["zoom"] > z0
    s = page.evaluate("([x, y]) => window.__lab.imgToClient(x - 0.5, y - 0.5)", [p0["fx"], p0["fy"]])
    assert abs(s["x"] - cx) <= 1 and abs(s["y"] - cy) <= 1, (s, cx, cy)
    # Ctrl + left drag pans in any tool: the Brush paints nothing
    pick_class(page, 0 if int(arr[y, x]) != 0 else 1)
    page.keyboard.press("b")
    cx, cy = centre(page)
    a = state(page)
    page.mouse.move(cx, cy)
    page.keyboard.down("Control"); page.mouse.down()
    page.mouse.move(cx - 50, cy - 20, steps=6)
    page.mouse.up(); page.keyboard.up("Control")
    b = state(page)
    assert b["zoom"] == a["zoom"] and not b["dirty"]
    assert abs((a["vx"] - b["vx"]) - 50) <= 1 and abs((a["vy"] - b["vy"]) - 20) <= 1, (a, b)
    assert (mask_array(stem) == arr).all()
    assert not page.errors


def test_double_click_toggles_fit_and_4x(page, frames):
    open_editor(page, frames[2]["stem"])
    cx, cy = centre(page)
    fit = state(page)["zoom"]
    page.mouse.dblclick(cx, cy)
    assert abs(state(page)["zoom"] - 4) < 0.01
    page.wait_for_timeout(400)
    page.mouse.dblclick(cx, cy)
    assert abs(state(page)["zoom"] - fit) < 0.01
    assert not page.errors


SWIPE = """async (a) => { const c = document.querySelector('[data-testid="editor-canvas"]');
  const f = (t, x, y) => c.dispatchEvent(new PointerEvent(t, { pointerId: 7, pointerType: 'touch', isPrimary: true, button: 0, clientX: x, clientY: y, bubbles: true, cancelable: true }));
  f('pointerdown', a.x, a.y);
  for (let i = 1; i <= 6; i++) { await new Promise((r) => setTimeout(r, 10)); f('pointermove', a.x + a.dx * i / 6, a.y + a.dy * i / 6); }
  f('pointerup', a.x + a.dx, a.y + a.dy); }"""


def test_touch_fling_glides_then_stops(touch_page, frames):
    open_touch(touch_page, frames[3]["stem"])
    mx, my = canvas_mid(touch_page)
    fit = state(touch_page)["zoom"]
    pinch(touch_page, mx, my, 20, 120)
    assert state(touch_page)["zoom"] > fit * 2
    v0 = state(touch_page)
    touch_page.evaluate(SWIPE, {"x": mx - 30, "y": my, "dx": 60, "dy": 0})
    lift = state(touch_page)["vx"]
    assert lift - v0["vx"] >= 59 and state(touch_page)["selection"] is None
    touch_page.wait_for_timeout(120)
    mid = state(touch_page)["vx"]
    assert mid > lift + 3, (lift, mid)                                  # still gliding after the finger left
    last = None
    for _ in range(40):
        touch_page.wait_for_timeout(150)
        cur = state(touch_page)["vx"]
        if last is not None and abs(cur - last) < 0.05:
            break
        last = cur
    else:
        raise AssertionError("fling never stopped")
    end = state(touch_page)["vx"]
    touch_page.wait_for_timeout(300)
    assert abs(state(touch_page)["vx"] - end) < 0.01 and end > mid
    # a new pointer down stops a running fling
    touch_page.evaluate(SWIPE, {"x": mx - 30, "y": my, "dx": -80, "dy": 0})
    touch_page.wait_for_timeout(40)
    fire(touch_page, "pointerdown", 9, mx, my)
    held = state(touch_page)["vx"]
    touch_page.wait_for_timeout(200)
    assert abs(state(touch_page)["vx"] - held) < 0.01
    fire(touch_page, "pointerup", 9, mx, my)
    assert not touch_page.errors


def test_pinch_leaves_no_fling(touch_page, frames):
    """A pinch moves the view through zoom; lifting it must not fling (v26.1.4 review fix)."""
    open_touch(touch_page, frames[3]["stem"])
    mx, my = canvas_mid(touch_page)
    pinch(touch_page, mx, my, 20, 120)
    s0 = state(touch_page)
    touch_page.wait_for_timeout(300)
    s1 = state(touch_page)
    assert abs(s1["vx"] - s0["vx"]) < 0.01 and abs(s1["vy"] - s0["vy"]) < 0.01, (s0, s1)
    assert not touch_page.errors
