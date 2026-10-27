from conftest import BASE, api, tid


def open_browse(page):
    page.goto(f"{BASE}/#browse")
    page.wait_for_selector(tid("frame-card"))


def test_browse_filter_and_tag(page, frames, snow_stems):
    open_browse(page)
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
    stems = {c.get_attribute("data-stem") for c in page.locator(tid("frame-card")).all()}
    assert stems == set(snow_stems)
    # tag: select the first card, press 2 (fix), then 4 (todo)
    first = page.locator(tid("frame-card")).first
    stem = first.get_attribute("data-stem")
    first.click()
    page.keyboard.press("2")
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'fix'")
    assert api(f"/api/frames/{stem}")["status"] == "fix"
    page.keyboard.press("4")
    page.wait_for_function(f"() => document.querySelector('[data-stem=\"{stem}\"]').dataset.status === 'todo'")
    assert api(f"/api/frames/{stem}")["status"] == "todo"
    assert not page.errors
