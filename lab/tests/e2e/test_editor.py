"""Editor: zoom, 1 px brush, undo/redo, save, backup + revert, region relabel, polygon, protect."""
import numpy as np
import cv2
from conftest import (SNOW, api, click_pixel, drag_pixels, find_pixel, mask_array, open_editor, pick_class, pixel, state, tid,
                      to_client, zoom_to)


def test_brush_undo_redo_save_revert(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    open_editor(page, stem)
    pick_class(page, SNOW)
    page.keyboard.press("b")
    page.fill(tid("brush-size"), "1")
    page.select_option(tid("brush-shape"), "square")
    zoom_to(page, x, y, 8.0)
    assert state(page)["zoom"] >= 8
    assert pixel(page, x, y) == old
    click_pixel(page, x, y)
    assert pixel(page, x, y) == SNOW
    assert pixel(page, x + 1, y) == old and pixel(page, x, y + 1) == old      # exactly one pixel
    page.keyboard.press("Control+z")
    assert pixel(page, x, y) == old
    page.keyboard.press("Control+Shift+z")
    assert pixel(page, x, y) == SNOW
    assert state(page)["dirty"]
    # no backup before the first save
    try:
        api(f"/api/frames/{stem}/revert", "POST")
        raise RuntimeError("revert before any save must 404")
    except AssertionError as e:
        assert "404" in str(e)
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    assert after[y, x] == SNOW
    assert int((after != before).sum()) == 1
    assert api(f"/api/frames/{stem}")["status"] == "edited"
    # revert through the UI (confirm is auto-accepted)
    page.click(tid("btn-revert"))
    page.wait_for_function(f"() => window.__lab.getPixel({x}, {y}) === {old}")
    assert (mask_array(stem) == before).all()
    assert api(f"/api/frames/{stem}")["status"] == "todo"
    assert not page.errors


def test_region_relabel(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    open_editor(page, stem)
    pick_class(page, SNOW)
    page.keyboard.press("r")
    click_pixel(page, x, y)
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    _, lbl = cv2.connectedComponents((before == old).astype(np.uint8), connectivity=4)
    comp = lbl == lbl[y, x]
    assert (after[comp] == SNOW).all()
    assert (after[~comp] == before[~comp]).all()


def test_polygon(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    open_editor(page, stem)
    pick_class(page, SNOW)
    page.keyboard.press("l")
    cx, cy = 640, 360
    for dx, dy in ((-80, -60), (80, -60), (80, 60), (-80, 60)):
        click_pixel(page, cx + dx, cy + dy)
    page.keyboard.press("Enter")
    assert pixel(page, cx, cy) == SNOW
    assert pixel(page, cx + 200, cy) == before[cy, cx + 200]
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    inside = after[cy - 55:cy + 55, cx - 75:cx + 75]
    assert (inside == SNOW).all()
    assert (after[:cy - 62] == before[:cy - 62]).all()


def test_protect_mode(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    # a band of the frame holding several classes; protect the most frequent one in it
    band = before[300:420, 300:980]
    vals, cnt = np.unique(band, return_counts=True)
    assert len(vals) >= 2
    protect = int(vals[np.argmax(cnt)])
    target = SNOW if protect != SNOW else 0
    open_editor(page, stem)
    pick_class(page, target)
    page.select_option(tid("protect-select"), str(protect))
    page.keyboard.press("b")
    page.fill(tid("brush-size"), "64")
    drag_pixels(page, [(300, 360), (980, 360)])
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    changed = after != before
    assert changed.any()
    assert (before[changed] == protect).all()          # only protected-class pixels changed
    assert (after[changed] == target).all()
