"""v26.1.4 (end of file): clear the class, selected-class-only mask, numbered detections.
v26.1.3: no Work group, no inline vertical scroll, Detected / Undetected classes with counts, class only when picked."""
import cv2
import numpy as np
from conftest import SNOW, W, H, api, click_pixel, find_pixel, mask_array, open_editor, pick_class, pixel, put_mask, state, tid, to_client


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


def mask_px(page, x, y):
    return page.evaluate("([x, y]) => window.__lab.maskPixel(x, y)", [x, y])


def class_color(cid):
    return next(c["color"] for c in api("/api/taxonomy")["classes"] if c["id"] == cid)


def components(arr, cid):
    """[(px, [x0, y0, x1, y1])] of the 4-connected regions of class cid, largest first."""
    n, _, st, _ = cv2.connectedComponentsWithStats((arr == cid).astype(np.uint8), connectivity=4)
    out = [(int(st[i, 4]), [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2] - 1), int(st[i, 1] + st[i, 3] - 1)]) for i in range(1, n)]
    return sorted(out, key=lambda t: -t[0])


def test_clear_button_and_page_escape(page, clean_frame):
    open_editor(page, clean_frame())
    ac = page.locator(tid("active-class"))
    pick_class(page, SNOW)
    assert ac.is_visible()
    page.click(tid("active-class-clear"))
    assert not ac.is_visible() and state(page)["cls"] is None
    pick_class(page, SNOW)
    page.keyboard.press("Escape")                            # focus is not in an input, nothing pending: Esc clears the class
    assert state(page)["cls"] is None and not ac.is_visible()
    # Esc order: the Inspect selection first, then the class
    pick_class(page, SNOW)
    x, y, _ = find_pixel(mask_array(state(page)["stem"]))
    click_pixel(page, x, y)
    assert state(page)["selection"] is not None
    page.keyboard.press("Escape")
    assert state(page)["selection"] is None and state(page)["cls"] == SNOW
    page.keyboard.press("Escape")
    assert state(page)["cls"] is None
    assert not page.errors


def test_selected_class_only_mask(page, clean_frame):
    stem = clean_frame()
    arr = mask_array(stem)
    ids, cnt = np.unique(arr, return_counts=True)
    a = int(ids[cnt.argmax()])
    ys, xs = np.nonzero(arr == a); xa, ya = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    ys, xs = np.nonzero(arr != a); xb, yb = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    open_editor(page, stem)
    assert mask_px(page, xa, ya)[3] == 255 and mask_px(page, xb, yb)[3] == 255     # no class: every class is shown
    pick_class(page, a)
    assert mask_px(page, xb, yb) == [0, 0, 0, 0]
    assert mask_px(page, xa, ya) == [*class_color(a), 255]
    page.keyboard.press("o")                                  # outline mode outlines only that class
    assert mask_px(page, xb, yb) == [0, 0, 0, 0]
    page.keyboard.press("o")
    assert mask_px(page, xa, ya) == [*class_color(a), 255]
    page.click(tid("active-class-clear"))
    assert mask_px(page, xb, yb)[3] == 255 and mask_px(page, xa, ya)[3] == 255
    assert not page.errors


def test_detections_match_connected_components(page, clean_frame):
    stem = clean_frame()
    arr = mask_array(stem)
    ids, cnt = np.unique(arr, return_counts=True)
    cid = int(ids[cnt.argmax()])
    comps = components(arr, cid)
    open_editor(page, stem)
    det = page.locator(tid("detections"))
    assert not det.is_visible()
    pick_class(page, cid)
    assert det.is_visible()
    st = state(page)["detections"]
    assert [d["px"] for d in st] == [c[0] for c in comps] and [d["k"] for d in st] == list(range(1, len(comps) + 1))
    for d, (px, bbox) in zip(st, comps):
        if sum(1 for c in comps if c[0] == px) == 1:
            assert d["bbox"] == bbox
        assert arr[d["seed"][1], d["seed"][0]] == cid
    chips = page.locator(tid("detection-item"))
    shown = min(60, len(comps))
    assert chips.count() == shown
    for i in range(shown):
        assert chips.nth(i).get_attribute("data-k") == str(i + 1) and chips.nth(i).inner_text() == f"#{i + 1} {comps[i][0]} px"
    assert f"Detections ({len(comps)})" in det.inner_text()
    # chip #1: selects the largest region, centred and inside the view
    chips.first.click()
    sel = state(page)["selection"]
    assert sel["px"] == comps[0][0] and sel["id"] == cid
    assert "act" in chips.first.get_attribute("class")
    cb = page.locator(tid("editor-canvas")).bounding_box()
    x0, y0, x1, y1 = sel["bbox"]
    a, b = to_client(page, x0 - 0.5, y0 - 0.5), to_client(page, x1 + 0.5, y1 + 0.5)
    assert a["x"] >= cb["x"] - 1 and a["y"] >= cb["y"] - 1 and b["x"] <= cb["x"] + cb["width"] + 1 and b["y"] <= cb["y"] + cb["height"] + 1, (a, b, cb)
    # a region picked on the image highlights its own chip
    if len(comps) > 1:
        px2, bb2 = comps[1]
        k = 2
        page.locator(tid("detection-item")).nth(1).click()
        assert state(page)["selection"]["px"] == px2
        assert "act" in page.locator(tid("detection-item")).nth(1).get_attribute("class") and "act" not in chips.first.get_attribute("class")
    page.keyboard.press("Escape")
    assert state(page)["selection"] is None and page.locator(f'{tid("detection-item")}.act').count() == 0
    page.keyboard.press("Escape")
    assert not det.is_visible() and state(page)["detections"] is None
    assert not page.errors


def test_detections_absent_class_and_more_button(page, clean_frame):
    stem = clean_frame()
    arr = np.zeros((H, W), np.uint8)
    arr[:] = 1
    for i in range(70):                                       # 70 separate 3x3 snow regions of different sizes
        y, x = 20 + (i // 14) * 40, 20 + (i % 14) * 80
        arr[y:y + 3 + i % 4, x:x + 3] = SNOW
    put_mask(stem, arr)
    open_editor(page, stem)
    absent = next(c["id"] for c in api("/api/taxonomy")["classes"] if c["id"] not in (1, SNOW))
    pick_class(page, absent)
    assert page.locator(tid("detections")).inner_text().count("0 detections in this frame") == 1
    assert page.locator(tid("detection-item")).count() == 0
    pick_class(page, SNOW)
    chips = page.locator(tid("detection-item"))
    assert chips.count() == 60
    more = page.locator(tid("detections-more"))
    assert more.inner_text() == "+10 more"
    more.click()
    assert chips.count() == 70 and more.inner_text() == "less"
    more.click()
    assert chips.count() == 60
    px = [int(t.split()[1]) for t in chips.all_inner_texts()]
    assert px == sorted(px, reverse=True)
    assert not page.errors
