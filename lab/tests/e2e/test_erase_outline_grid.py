"""v26.1.7: Erase paints void unlabeled, Outline draws over the fill, the grid shows from 4x, class names are readable."""
import numpy as np
from conftest import api, click_pixel, find_pixel, mask_array, open_editor, pick_class, pixel, state, tid, zoom_to

VOID = 123


def _one_px_tool(page, key):
    page.keyboard.press(key)
    page.fill(tid("brush-size"), "1")
    page.select_option(tid("brush-shape"), "square")


def _two_classes(arr):
    x, y, a = find_pixel(arr, not_class=VOID)
    x2, y2, b = find_pixel(arr, not_class=a)
    return (x, y, a), (x2, y2, b)


def test_erase_paints_void_without_a_class(page, clean_frame):
    stem = clean_frame()
    (x, y, a), _ = _two_classes(mask_array(stem))
    open_editor(page, stem)
    assert state(page)["cls"] is None
    _one_px_tool(page, "e")
    zoom_to(page, x, y, 8.0)
    assert state(page)["eraseTo"] == VOID
    click_pixel(page, x, y)
    assert pixel(page, x, y) == VOID and pixel(page, x + 1, y) == a
    assert page.inner_text("#sb-tool") == "Erase 1s \u2192 void unlabeled"
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    assert mask_array(stem)[y, x] == VOID
    tools = api(f"/api/frames/{stem}")
    assert tools["status"] == "edited"
    assert not page.errors


def test_erase_respects_protect(page, clean_frame):
    stem = clean_frame()
    (x, y, a), (x2, y2, b) = _two_classes(mask_array(stem))
    open_editor(page, stem)
    page.select_option(tid("protect-select"), str(a))
    _one_px_tool(page, "e")
    zoom_to(page, x2, y2, 8.0)
    click_pixel(page, x2, y2)
    assert pixel(page, x2, y2) == b                       # not the protected class: untouched
    page.keyboard.press("f")                              # back to fit, then zoom on the protected class
    zoom_to(page, x, y, 8.0)
    click_pixel(page, x, y)
    assert pixel(page, x, y) == VOID
    assert not page.errors


def test_x_and_right_click_do_nothing(page, clean_frame):
    stem = clean_frame()
    (x, y, a), _ = _two_classes(mask_array(stem))
    open_editor(page, stem)
    page.keyboard.press("x")
    assert state(page)["cls"] is None and not state(page)["dirty"]
    pick_class(page, a)
    page.locator(f'{tid("picker-item")}[data-id="{a}"]').first.click(button="right")
    page.locator(f'{tid("picker-item")}[data-id="{a}"]').first.click()   # clears the class again
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("x")
    assert state(page)["cls"] is None
    assert page.locator(".pk-item.sec").count() == 0
    _one_px_tool(page, "e")                               # Erase does not depend on any picked class
    zoom_to(page, x, y, 8.0)
    click_pixel(page, x, y)
    assert pixel(page, x, y) == VOID
    assert not page.errors


def test_outline_keeps_the_fill_and_adds_lines(page, clean_frame):
    stem = clean_frame()
    arr = mask_array(stem)
    x, y, _ = find_pixel(arr)
    ys, xs = np.nonzero(arr[:, :-1] != arr[:, 1:])
    bx, by = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    open_editor(page, stem)
    fill = lambda px, py: page.evaluate("([x, y]) => window.__lab.maskPixel(x, y)", [px, py])
    line = lambda px, py: page.evaluate("([x, y]) => window.__lab.linePixel(x, y)", [px, py])
    f0, fb = fill(x, y), fill(bx, by)
    assert f0[3] == 255 and line(bx, by)[3] == 0
    page.keyboard.press("o")
    assert state(page)["outline"]
    assert fill(x, y) == f0 and fill(bx, by) == fb        # the fill is untouched
    assert line(x, y)[3] == 0                             # interior: no line
    assert line(bx, by)[3] == 255 and line(bx, by)[:3] == fb[:3]
    page.keyboard.press("o")
    assert fill(x, y) == f0 and line(bx, by)[3] == 0
    assert not page.errors


def test_outline_with_a_class_covers_only_that_class(page, clean_frame):
    stem = clean_frame()
    arr = mask_array(stem)
    ys, xs = np.nonzero(arr[:, :-1] != arr[:, 1:])
    bx, by = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    a, b = int(arr[by, bx]), int(arr[by, bx + 1])
    open_editor(page, stem)
    pick_class(page, a)
    page.keyboard.press("o")
    line = lambda px, py: page.evaluate("([x, y]) => window.__lab.linePixel(x, y)", [px, py])
    assert line(bx, by)[3] == 255 and line(bx + 1, by)[3] == 0
    assert page.evaluate("([x, y]) => window.__lab.maskPixel(x, y)", [bx, by])[3] == 255
    assert not page.errors


def test_grid_draws_from_4x(page, clean_frame):
    stem = clean_frame()
    open_editor(page, stem)
    x, y, a = find_pixel(mask_array(stem))
    page.wait_for_timeout(150)
    assert state(page)["zoom"] < 4 and not state(page)["gridDrawn"]
    zoom_to(page, x, y, 4.0)
    page.wait_for_function("() => window.__lab.state().gridDrawn")
    page.keyboard.press("o")                              # still drawn with Outline on and a class selected
    pick_class(page, a)
    page.keyboard.press("g")
    page.wait_for_function("() => !window.__lab.state().gridDrawn")
    page.keyboard.press("g")
    page.wait_for_function("() => window.__lab.state().gridDrawn")
    assert not page.errors


def test_no_picker_chip_shows_a_raw_name(page, frames):
    open_editor(page, frames[0]["stem"])
    texts = page.eval_on_selector_all(tid("picker-item"), "els => els.map(e => e.textContent)")
    assert len(texts) >= 124 and not [t for t in texts if "--" in t]
    names = [c["name"] for c in api("/api/taxonomy")["classes"]]
    assert len(names) == len(set(names))
