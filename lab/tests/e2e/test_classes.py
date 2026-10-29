"""v26.1.3: no Work group, no inline vertical scroll, Detected / Undetected classes with counts, class only when picked."""
import cv2
import numpy as np
from conftest import SNOW, api, click_pixel, find_pixel, mask_array, open_editor, pick_class, pixel, state, tid, to_client


def expected(stem):
    """{class id: (4-connected regions, px)} for the frame's mask on disk."""
    arr = mask_array(stem)
    out = {}
    for c in np.unique(arr):
        n, _ = cv2.connectedComponents((arr == c).astype(np.uint8), connectivity=4)
        out[int(c)] = (n - 1, int((arr == c).sum()))
    return out


def chip_ids(page, group):
    return [int(v) for v in page.eval_on_selector_all(f'{tid("group-" + group)} {tid("picker-item")}', "els => els.map(e => e.dataset.id)")]


def test_work_group_gone_and_panes_direct(page, frames):
    open_editor(page, frames[0]["stem"])
    assert page.locator('[data-gid="work"]').count() == 0
    assert page.locator("#workTitle").count() == 0
    assert page.evaluate("[...document.querySelectorAll('.pane')].every(p => p.parentElement.id === 'page')")
    assert page.evaluate("document.querySelectorAll('#page > .pane').length") == 3


def test_active_class_only_when_selected(page, clean_frame):
    open_editor(page, clean_frame())
    ac = page.locator(tid("active-class"))
    assert not ac.is_visible() and state(page)["cls"] is None
    chip = page.locator(f'{tid("picker-item")}[data-id="{SNOW}"]')
    page.fill(tid("picker-search"), str(SNOW))
    chip.first.click()
    assert ac.is_visible() and state(page)["cls"] == SNOW
    assert next(c["name"] for c in api("/api/taxonomy")["classes"] if c["id"] == SNOW) in ac.inner_text() and f"({SNOW})" in ac.inner_text()
    chip.first.click()                                       # the active chip again clears it
    assert not ac.is_visible() and state(page)["cls"] is None
    chip.first.click()
    assert ac.is_visible()
    page.fill(tid("picker-search"), "")
    page.focus(tid("picker-search"))
    page.keyboard.press("Escape")                            # empty search box: Esc clears
    assert not ac.is_visible() and state(page)["cls"] is None
    assert not page.errors


def test_detected_undetected_partition_and_counts(page, clean_frame):
    stem = clean_frame()
    exp = expected(stem)
    open_editor(page, stem)
    tax = [c["id"] for c in api("/api/taxonomy")["classes"]]
    det = chip_ids(page, "edit-detected")
    undet = chip_ids(page, "edit-undetected")
    assert set(det) == set(exp)
    assert set(undet) == set(tax) - set(exp)
    assert sorted(det + undet) == sorted(tax)                # covers the taxonomy exactly once
    counts = state(page)["classCounts"]
    assert {int(k) for k in counts} == set(exp)
    for cid, (n, px) in exp.items():
        assert counts[str(cid)] == {"items": n, "px": px}, cid
        chip = page.locator(f'{tid("group-edit-detected")} {tid("picker-item")}[data-id="{cid}"]')
        assert chip.locator(".muted.mono").inner_text() == str(n)
        assert f"{n} detections, {px} px" in chip.get_attribute("title")
    assert [exp[i][0] for i in det] == sorted((exp[i][0] for i in det), reverse=True)    # most detections first
    assert page.locator(f'{tid("group-edit-detected")} > .gh small').inner_text() == str(len(det))
    assert page.locator(f'{tid("group-edit-undetected")} > .gh small').inner_text() == str(len(undet))
    assert page.locator(f'{tid("group-edit-undetected")} {tid("picker-item")} .cnt:visible').count() == 0
    # undetected chips sit in collapsed hier groups
    assert page.locator(f'{tid("group-edit-undetected")} [data-testid^="group-hier-"][data-collapsed="true"]').count() > 0
    assert not page.errors


def test_paint_moves_class_and_undo_moves_back(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    present = {int(v) for v in np.unique(before)}
    new = next(c["id"] for c in api("/api/taxonomy")["classes"] if c["id"] not in present)
    x, y, _ = find_pixel(before, not_class=new)
    open_editor(page, stem)
    assert new in chip_ids(page, "edit-undetected")
    pick_class(page, new)
    page.keyboard.press("r")
    click_pixel(page, x, y)
    assert pixel(page, x, y) == new
    assert new in chip_ids(page, "edit-detected") and new not in chip_ids(page, "edit-undetected")
    assert state(page)["classCounts"][str(new)]["items"] == 1
    chip = page.locator(f'{tid("group-edit-detected")} {tid("picker-item")}[data-id="{new}"]')
    assert chip.locator(".muted.mono").inner_text() == "1"
    page.keyboard.press("Control+z")
    assert new in chip_ids(page, "edit-undetected") and new not in chip_ids(page, "edit-detected")
    assert str(new) not in state(page)["classCounts"]
    page.keyboard.press("Control+Shift+z")
    assert new in chip_ids(page, "edit-detected")
    assert not page.errors


def test_painting_without_a_class_does_nothing(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before)
    open_editor(page, stem)
    assert state(page)["cls"] is None
    for key in ("b", "r"):
        page.keyboard.press(key)
        click_pixel(page, x, y)
        assert pixel(page, x, y) == old
    assert page.locator(".toast", has_text="Pick a class first").count() >= 1
    assert not state(page)["dirty"]
    # one stroke with many moves toasts once at most per press
    page.keyboard.press("b")
    page.evaluate("document.querySelectorAll('.toast').forEach(t => t.remove())")
    c = to_client(page, x, y)
    page.mouse.move(c["x"], c["y"]); page.mouse.down(); page.mouse.move(c["x"] + 60, c["y"] + 10, steps=15); page.mouse.up()
    assert page.locator(".toast", has_text="Pick a class first").count() == 1
    assert not state(page)["dirty"]
    # Inspect still works without a class
    page.keyboard.press("v")
    click_pixel(page, x, y)
    assert state(page)["selection"]["id"] == old
    assert not page.errors


def no_vertical_scrollbars(page):
    return page.evaluate("""() => [...document.querySelectorAll('#page *')].filter(e => {
        const s = getComputedStyle(e);
        return (s.overflowY === 'auto' || s.overflowY === 'scroll') && e.scrollHeight > e.clientHeight + 1 && e.offsetParent !== null;
      }).map(e => e.id || e.className || e.tagName)""")


def test_no_inline_vertical_scroll(page, clean_frame):
    open_editor(page, clean_frame())
    assert page.evaluate("getComputedStyle(document.querySelector('#page')).overflowY") == "auto"
    assert no_vertical_scrollbars(page) == []
    page.click(tid("tab-taxonomy"))
    page.wait_for_selector(tid("class-row"))
    assert no_vertical_scrollbars(page) == []
    # the whole table is laid out, not clipped
    assert page.evaluate("document.querySelector('#tTable').scrollHeight <= document.querySelector('#tTable').clientHeight + 1")
    assert not page.errors
