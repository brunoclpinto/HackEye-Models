import time

from conftest import BASE, api, open_editor, tid


def open_page(page):
    page.goto(f"{BASE}/#edit")                  # default route: Edit, first frame of the filter
    page.wait_for_selector(tid("nav-pos"))
    page.wait_for_function("() => window.__lab && window.__lab.state().stem")


def cur(page):
    return page.evaluate("window.__lab.state().stem")


def at(page, stem):
    page.wait_for_function(f"() => window.__lab.state().stem === '{stem}'")


def pos(page):
    el = page.locator(tid("nav-pos"))
    return int(el.get_attribute("data-index")), int(el.get_attribute("data-total"))


def wait_status(stem, status):
    for _ in range(50):
        if api(f"/api/frames/{stem}")["status"] == status:
            return
        time.sleep(0.1)
    raise AssertionError(f"{stem} never became {status}")


def test_browse_filter_and_tag(page, frames, snow_stems):
    open_page(page)
    assert page.locator(tid("frame-count")).inner_text().strip().startswith("20")
    assert pos(page) == (1, 20)
    # split filter
    nval = api("/api/frames?split=val&limit=1000")["total"]
    assert 0 < nval < 20
    page.select_option(tid("filter-split"), "val")
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"frame-count\"]').textContent.trim().startsWith('{nval}')")
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.total === '{nval}'")
    for _ in range(50):                                                 # the view moves into the filter
        if api(f"/api/frames/{cur(page)}")["split"] == "val":
            break
        time.sleep(0.1)
    assert api(f"/api/frames/{cur(page)}")["split"] == "val"
    page.select_option(tid("filter-split"), "")
    # class filter: nature snow (id 62) -> the 3 snow frames; the view jumps to the first one
    page.select_option(tid("filter-class"), "62")
    n = len(snow_stems)
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"frame-count\"]').textContent.trim().startsWith('{n}')")
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.total === '{n}'")
    at(page, snow_stems[0])
    assert pos(page) == (1, n)
    # tag the current frame: Shift+2 = fix, then the buttons, then Shift+4 = todo
    stem = cur(page)
    page.keyboard.press("Shift+2")
    wait_status(stem, "fix")
    page.click(tid("tag-ok"))
    wait_status(stem, "ok")
    page.click(tid("tag-todo"))
    wait_status(stem, "todo")
    page.keyboard.press("Shift+2")
    wait_status(stem, "fix")
    page.keyboard.press("Shift+4")
    wait_status(stem, "todo")
    assert not page.errors


def test_next_prev_step_in_order(page, frames):
    s = [r["stem"] for r in frames]
    open_editor(page, s[3])
    page.wait_for_selector(tid("nav-pos"))
    page.wait_for_function("() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.total === '20'")
    assert pos(page) == (4, 20)
    page.keyboard.press("ArrowRight"); at(page, s[4])
    page.click(tid("nav-next")); at(page, s[5])
    page.keyboard.press("n"); at(page, s[6])
    page.keyboard.press("ArrowLeft"); at(page, s[5])
    page.click(tid("nav-prev")); at(page, s[4])
    page.keyboard.press("p"); at(page, s[3])
    page.wait_for_function("() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.index === '4'")
    assert not page.errors


def test_next_wraps_last_to_first_and_prev_first_to_last(page, frames):
    s = [r["stem"] for r in frames]
    # whole list: Next on the last frame -> first
    open_editor(page, s[-1])
    page.wait_for_function("() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.index === '20'")
    page.click(tid("nav-next"))
    at(page, s[0])
    page.wait_for_function("() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.index === '1'")
    # Prev on the first frame -> last
    page.click(tid("nav-prev"))
    at(page, s[-1])
    assert pos(page) == (20, 20)
    # same with the keys
    page.keyboard.press("ArrowRight"); at(page, s[0])
    page.keyboard.press("ArrowLeft"); at(page, s[-1])
    page.keyboard.press("n"); at(page, s[0])
    page.keyboard.press("p"); at(page, s[-1])
    # inside a filter (snow frames only) the wrap stays within the filter
    snow = [r["stem"] for r in api("/api/frames?cls=62&minpx=2000&limit=1000")["rows"]]
    assert len(snow) >= 3
    page.goto(f"{BASE}/#edit/{snow[-1]}?cls=62")
    page.wait_for_function(f"() => window.__lab.state().stem === '{snow[-1]}'")
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"nav-pos\"]').dataset.total === '{len(snow)}'")
    page.click(tid("nav-next")); at(page, snow[0])
    page.click(tid("nav-prev")); at(page, snow[-1])
    assert not page.errors


def test_group_collapse_persists_after_reload(page, frames):
    open_page(page)
    grp = page.locator(tid("group-browse"))
    assert grp.get_attribute("data-collapsed") == "false"
    grp.locator(tid("group-toggle")).first.click()
    assert grp.get_attribute("data-collapsed") == "true"
    assert not page.locator(tid("filter-split")).is_visible()          # the filter bar folds away; the nav row stays in the header
    assert page.locator(tid("nav-next")).is_visible()
    page.reload()
    page.wait_for_selector(tid("group-browse"))
    assert page.locator(tid("group-browse")).get_attribute("data-collapsed") == "true"
    assert page.locator(tid("group-image")).get_attribute("data-collapsed") == "false"     # others unchanged
    page.locator(tid("group-browse")).locator(tid("group-toggle")).first.click()
    assert page.locator(tid("group-browse")).get_attribute("data-collapsed") == "false"
    assert page.locator(tid("filter-split")).is_visible()
