from conftest import BASE, api, open_editor, state, tid


def open_page(page):
    page.goto(f"{BASE}/#edit")                  # default route: Edit, first frame of the filter
    page.wait_for_selector(tid("frame-card"))


def test_browse_filter_and_tag(page, frames, snow_stems):
    open_page(page)
    assert page.locator(tid("frame-count")).inner_text().strip().startswith("20")
    # split filter
    nval = api("/api/frames?split=val&limit=1000")["total"]
    assert 0 < nval < 20
    page.select_option(tid("filter-split"), "val")
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"frame-count\"]').textContent.trim().startsWith('{nval}')")
    assert {c.get_attribute("data-split") for c in page.locator(tid("frame-card")).all()} == {"val"}
    page.select_option(tid("filter-split"), "")
    # class filter: nature snow (id 62) -> the 3 snow frames
    page.select_option(tid("filter-class"), "62")
    n = len(snow_stems)
    page.wait_for_function(f"() => document.querySelector('[data-testid=\"frame-count\"]').textContent.trim().startsWith('{n}')")
    page.wait_for_function(f"() => document.querySelectorAll('[data-testid=\"frame-card\"]').length === {n}")
    stems = {c.get_attribute("data-stem") for c in page.locator(tid("frame-card")).all()}
    assert stems == set(snow_stems)
    # clicking a card opens that frame; Shift+2 tags it fix, the todo button sets it back
    first = page.locator(tid("frame-card")).first
    stem = first.get_attribute("data-stem")
    first.click()
    page.wait_for_function(f"() => window.__lab.state().stem === '{stem}'")
    page.keyboard.press("Shift+2")
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'fix'")
    assert api(f"/api/frames/{stem}")["status"] == "fix"
    page.click(tid("tag-todo"))
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'todo'")
    assert api(f"/api/frames/{stem}")["status"] == "todo"
    page.click(tid("tag-ok"))
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'ok'")
    page.keyboard.press("Shift+4")
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'todo'")
    assert api(f"/api/frames/{stem}")["status"] == "todo"
    assert not page.errors


def test_carousel_navigation(page, frames):
    s = [r["stem"] for r in frames]
    open_editor(page, s[3])
    page.wait_for_selector(tid("frame-card"))
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"() => window.__lab.state().stem === '{s[4]}'")
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(f"() => window.__lab.state().stem === '{s[3]}'")
    page.keyboard.press("End")
    page.wait_for_function(f"() => window.__lab.state().stem === '{s[-1]}'")
    page.keyboard.press("Home")
    page.wait_for_function(f"() => window.__lab.state().stem === '{s[0]}'")
    # the current frame's card is highlighted
    page.wait_for_selector(f'{tid("frame-card")}[data-stem="{s[0]}"].cur')
    # clicking another card opens it
    page.locator(f'{tid("frame-card")}[data-stem="{s[2]}"]').click()
    page.wait_for_function(f"() => window.__lab.state().stem === '{s[2]}'")
    assert not page.errors


def test_group_collapse_persists_after_reload(page, frames):
    open_page(page)
    grp = page.locator(tid("group-browse"))
    assert grp.get_attribute("data-collapsed") == "false"
    grp.locator(tid("group-toggle")).first.click()
    assert grp.get_attribute("data-collapsed") == "true"
    assert not page.locator(tid("carousel")).is_visible()
    page.reload()
    page.wait_for_selector(tid("group-browse"))
    assert page.locator(tid("group-browse")).get_attribute("data-collapsed") == "true"
    assert page.locator(tid("group-image")).get_attribute("data-collapsed") == "false"     # others unchanged
    page.locator(tid("group-browse")).locator(tid("group-toggle")).first.click()
    assert page.locator(tid("group-browse")).get_attribute("data-collapsed") == "false"
    page.wait_for_selector(tid("frame-card"))
