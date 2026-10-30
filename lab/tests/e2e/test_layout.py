"""v26.1.6 layout: Filters / Frame titles, Frame header (note, tags, Undo/Redo, nav, Save), side column, hover titles, removed controls."""
from conftest import BASE, SNOW, api, click_pixel, find_pixel, mask_array, open_editor, pick_class, state, tid

VISIBLE_NO_TITLE = """() => [...document.querySelectorAll('button, select, input[type=range], input[type=color], #tabs a')]
  .filter((e) => e.offsetParent !== null && !(e.title || '').trim())
  .map((e) => e.id || e.dataset.testid || e.textContent.trim().slice(0, 20) || e.tagName)"""


def box(page, sel):
    return page.locator(sel).bounding_box()


def test_titles_and_header(page, frames):
    open_editor(page, frames[0]["stem"])
    assert page.locator('[data-gid="browse"] > .gh .gt span').inner_text() == "Filters"
    assert page.locator('[data-gid="image"] > .gh .gt span').inner_text() == "Frame"
    head = page.locator('[data-gid="image"] > .gh')
    for t in ("frame-note", "tag-ok", "tag-fix", "tag-drop", "tag-todo", "btn-undo", "btn-redo", "nav-prev", "nav-pos", "nav-next", "btn-save"):
        assert head.locator(tid(t)).count() == 1, t
    assert page.locator('[data-gid="browse"] > .gh').locator(tid("nav-next")).count() == 0
    assert page.locator('[data-gid="browse"] > .gh').locator(tid("tag-ok")).count() == 0
    # the current status is highlighted on the header tags
    page.click(tid("tag-ok"))
    page.wait_for_function("() => document.querySelector('[data-testid=tag-ok]').classList.contains('on')")
    page.click(tid("tag-todo"))                              # leave the shared fixture frame as it was
    page.wait_for_function("() => document.querySelector('[data-testid=tag-todo]').classList.contains('on')")


def test_removed_controls(page, frames):
    open_editor(page, frames[0]["stem"])
    assert page.locator("#ePickerExtra").count() == 0
    assert page.locator('[data-testid="group-edit-frame"]').count() == 0
    for sel in ("#eStatus", "#eRevert", "#ePrev", "#eNext", tid("btn-revert")):
        assert page.locator(sel).count() == 0, sel


def test_save_button(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    open_editor(page, stem)
    assert page.locator(tid("btn-save")).is_disabled()
    pick_class(page, SNOW)
    page.keyboard.press("b")
    click_pixel(page, x, y)
    page.wait_for_function("() => window.__lab.state().dirty")
    assert page.locator(tid("btn-save")).is_enabled()
    page.click(tid("btn-save"))
    page.wait_for_function("() => !window.__lab.state().dirty")
    assert page.locator(tid("btn-save")).is_disabled()
    assert api(f"/api/frames/{stem}")["status"] == "edited"
    assert int((mask_array(stem) != before).sum()) > 0
    assert not page.errors


def test_undo_redo_buttons(page, clean_frame):
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=SNOW)
    open_editor(page, stem)
    assert page.locator(tid("btn-undo")).is_disabled() and page.locator(tid("btn-redo")).is_disabled()
    pick_class(page, SNOW)
    page.keyboard.press("b")
    click_pixel(page, x, y)
    page.wait_for_function("() => window.__lab.state().dirty")
    assert page.locator(tid("btn-undo")).is_enabled() and page.locator(tid("btn-redo")).is_disabled()
    page.click(tid("btn-undo"))
    page.wait_for_function("() => !window.__lab.state().dirty")
    assert page.locator(tid("btn-redo")).is_enabled()
    page.click(tid("btn-redo"))
    page.wait_for_function("() => window.__lab.state().dirty")
    assert page.locator(tid("btn-redo")).is_disabled()
    assert not page.errors


def test_note_saves_and_typing_is_not_a_shortcut(page, frames):
    stem = frames[1]["stem"]
    open_editor(page, stem)
    note = page.locator(tid("frame-note"))
    tool = state(page)["tool"]
    note.fill("bike lane edge")                              # contains b / e / r shortcut letters
    assert state(page)["tool"] == tool
    note.press("Enter")
    page.wait_for_function(f"() => fetch('/api/frames/{stem}').then(r => r.json()).then(j => j.note === 'bike lane edge')", polling=200)
    page.reload(); open_editor(page, stem)
    assert page.locator(tid("frame-note")).input_value() == "bike lane edge"
    note = page.locator(tid("frame-note")); note.fill(""); note.press("Enter")   # leave the shared fixture frame as it was
    page.wait_for_function(f"() => fetch('/api/frames/{stem}').then(r => r.json()).then(j => !j.note)", polling=200)
    assert not page.errors


def test_side_column(page, frames):
    page.set_viewport_size({"width": 1400, "height": 900})
    open_editor(page, frames[0]["stem"])
    side, body, canvas = box(page, "#eSide"), box(page, tid("editor")), box(page, tid("editor-canvas"))
    assert side["width"] <= body["width"] * 0.25 + 1, (side, body)
    assert side["x"] + side["width"] <= canvas["x"] + 1
    for g in ("edit-tools", "edit-protect", "edit-view"):
        assert page.locator(f'#eSide [data-gid="{g}"]').count() == 1
    w_edit = canvas["width"]
    page.click(tid("tab-taxonomy"))
    page.wait_for_function("() => document.querySelector('#eSide').hidden")
    page.wait_for_function(f"() => document.querySelector('[data-testid=editor-canvas]').getBoundingClientRect().width > {w_edit} + 50")
    page.click(tid("tab-edit"))
    page.wait_for_function("() => !document.querySelector('#eSide').hidden")
    page.wait_for_function(f"() => Math.abs(document.querySelector('[data-testid=editor-canvas]').getBoundingClientRect().width - {w_edit}) < 2")
    assert not page.errors


def test_side_column_stacks_on_phone(page, frames):
    page.set_viewport_size({"width": 390, "height": 844})
    open_editor(page, frames[0]["stem"])
    side, canvas = box(page, "#eSide"), box(page, tid("editor-canvas"))
    assert side["y"] + side["height"] <= canvas["y"] + 1, (side, canvas)
    assert side["width"] >= canvas["width"] - 20


def test_every_visible_button_has_a_title(page, frames):
    open_editor(page, frames[0]["stem"])
    page.wait_for_selector(tid("picker-item"), state="attached")
    assert page.evaluate(VISIBLE_NO_TITLE) == []
    page.click(tid("tab-taxonomy"))
    page.wait_for_selector(tid("class-row"))
    assert page.evaluate(VISIBLE_NO_TITLE) == []
    page.click(tid("btn-new-class"))
    page.wait_for_selector(tid("dlg-ok"))
    assert page.evaluate(VISIBLE_NO_TITLE) == []
    page.keyboard.press("Escape")
    page.click(tid("tab-rules"))
    page.wait_for_selector(tid("rule-from"))
    assert page.evaluate(VISIBLE_NO_TITLE) == []
    assert not page.errors
