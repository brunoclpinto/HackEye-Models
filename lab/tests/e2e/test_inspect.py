"""Inspect tool (default): click selects the 4-connected region of the class under the pointer; Esc clears; Enter relabels."""
import cv2
import numpy as np
from conftest import SNOW, find_pixel, click_pixel, mask_array, open_editor, pick_class, pixel, state, tid, zoom_to


def region_of(arr, x, y):
    cls = arr[y, x]
    _, lbl = cv2.connectedComponents((arr == cls).astype(np.uint8), connectivity=4)
    comp = lbl == lbl[y, x]
    ys, xs = np.nonzero(comp)
    return comp, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def test_inspect_is_default_and_selects_region(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    open_editor(page, stem)
    assert state(page)["tool"] == "inspect"
    assert "on" in (page.locator(tid("tool-inspect")).get_attribute("class") or "")
    zoom_to(page, x, y, 8.0)
    click_pixel(page, x, y)
    sel = state(page)["selection"]
    comp, bbox = region_of(before, x, y)
    assert sel["id"] == old == pixel(page, x, y)
    assert sel["px"] == int(comp.sum())
    assert sel["bbox"] == bbox
    label = page.locator(tid("selection-label")).inner_text()
    assert f"id {old}" in label and f"{int(comp.sum())} px" in label
    assert not state(page)["dirty"]                         # inspecting never edits
    page.keyboard.press("Escape")
    assert state(page)["selection"] is None
    assert not page.locator(tid("selection-label")).is_visible()
    assert not page.errors


def test_inspect_enter_relabels_with_undo_and_save(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    comp, _ = region_of(before, x, y)
    open_editor(page, stem)
    pick_class(page, SNOW)
    zoom_to(page, x, y, 8.0)
    click_pixel(page, x, y)
    assert state(page)["selection"]["id"] == old
    page.keyboard.press("Enter")
    assert state(page)["selection"] is None and state(page)["dirty"]
    assert pixel(page, x, y) == SNOW
    page.keyboard.press("Control+z")
    assert pixel(page, x, y) == old and not state(page)["dirty"]
    page.keyboard.press("Control+Shift+z")
    assert pixel(page, x, y) == SNOW
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    assert (after[comp] == SNOW).all()
    assert (after[~comp] == before[~comp]).all()


def test_inspect_enter_respects_protect(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    other = [int(v) for v in np.unique(before) if int(v) not in (old, SNOW)][0]
    open_editor(page, stem)
    pick_class(page, SNOW)
    page.select_option(tid("protect-select"), str(other))      # protect a class the selection does not contain
    zoom_to(page, x, y, 8.0)
    click_pixel(page, x, y)
    page.keyboard.press("Enter")
    assert pixel(page, x, y) == old and not state(page)["dirty"]
