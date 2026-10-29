"""v26.1.5: display-only view colour per class (View group) and label size slider."""
from conftest import api, click_pixel, find_pixel, mask_array, open_editor, pick_class, state, tid, zoom_to


def fresh(page, stem):
    open_editor(page, stem)
    page.evaluate("() => { try { localStorage.removeItem('lab.viewColors'); localStorage.removeItem('lab.labelSize'); } catch (e) {} }")
    page.reload()
    open_editor(page, stem)


def px(page, x, y):
    return page.evaluate("([x, y]) => window.__lab.maskPixel(x, y)", [x, y])


def sw_bg(page, cid):
    return page.eval_on_selector(f'{tid("picker-item")}[data-id="{cid}"] .sw', "e => getComputedStyle(e).backgroundColor")


def test_view_colour_changes_mask_and_chip_only(page, clean_frame):
    stem = clean_frame()
    x, y, cid = find_pixel(mask_array(stem))
    orig = next(c for c in api("/api/taxonomy")["classes"] if c["id"] == cid)["color"]
    fresh(page, stem)
    pick_class(page, cid)
    a0 = px(page, x, y)[3]
    page.fill(tid("view-color"), "#00ff66")
    page.wait_for_function("([x, y]) => window.__lab.maskPixel(x, y)[1] === 255", arg=[x, y])
    assert px(page, x, y) == [0, 255, 102, a0]
    assert sw_bg(page, cid) == "rgb(0, 255, 102)"
    assert "custom view colour" in page.locator(tid("view-color-hint")).inner_text()
    assert next(c for c in api("/api/taxonomy")["classes"] if c["id"] == cid)["color"] == orig
    page.reload()
    open_editor(page, stem)
    pick_class(page, cid)
    assert px(page, x, y)[:3] == [0, 255, 102]
    assert page.locator(tid("view-color-reset-all")).is_visible()
    page.click(tid("view-color-reset"))
    page.wait_for_function("([x, y, c]) => window.__lab.maskPixel(x, y)[0] === c", arg=[x, y, orig[0]])
    assert px(page, x, y) == [*orig, a0]
    assert not page.locator(tid("view-color-reset-all")).is_visible()
    assert not page.errors


def test_target_follows_selection_and_disables(page, clean_frame):
    stem = clean_frame()
    x, y, cid = find_pixel(mask_array(stem))
    fresh(page, stem)
    inp = page.locator(tid("view-color"))
    assert inp.is_disabled() and "Select a class or an item first" in (inp.get_attribute("title") or "")
    zoom_to(page, x, y, 4.0)
    click_pixel(page, x, y)
    assert state(page)["selection"]["id"] == cid
    assert inp.is_enabled()
    name = next(c["name"] for c in api("/api/taxonomy")["classes"] if c["id"] == cid)
    assert name in page.locator(tid("view-color-hint")).inner_text()
    page.fill(tid("view-color"), "#ff00ff")
    page.wait_for_function("([x, y]) => window.__lab.maskPixel(x, y)[0] === 255 && window.__lab.maskPixel(x, y)[2] === 255", arg=[x, y])
    page.keyboard.press("Escape")
    assert inp.is_disabled()
    assert not page.errors


def test_label_size_slider_grows_inspect_label(page, clean_frame):
    stem = clean_frame()
    x, y, _ = find_pixel(mask_array(stem))
    fresh(page, stem)
    zoom_to(page, x, y, 4.0)
    click_pixel(page, x, y)
    page.wait_for_function("() => window.__lab.state().labelBox")
    assert state(page)["labelSize"] == 13
    h13 = state(page)["labelBox"]["h"]
    page.fill(tid("label-size"), "26")
    page.wait_for_function("() => window.__lab.state().labelBox.h > 30")
    st = state(page)
    assert st["labelSize"] == 26 and st["labelBox"]["h"] > h13 * 1.8
    page.reload()
    open_editor(page, stem)
    assert state(page)["labelSize"] == 26
    assert not page.errors
