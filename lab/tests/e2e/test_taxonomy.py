from conftest import SNOW, api, click_pixel, find_pixel, mask_array, open_editor, pick_class, state, tid
from conftest import BASE


def test_create_class_and_paint(page, clean_frame):
    page.goto(f"{BASE}/#taxonomy")
    page.wait_for_selector(tid("class-row"))
    page.click(tid("btn-new-class"))
    page.fill(tid("dlg-name"), "e2e test class")
    page.fill(tid("dlg-hier"), "object")
    page.click(tid("dlg-ok"))
    page.wait_for_function("() => document.querySelector('[data-testid=\"class-row\"][data-custom=\"true\"]')")
    made = [c for c in api("/api/taxonomy")["classes"] if c["name"] == "e2e test class"]
    assert len(made) == 1 and made[0]["id"] >= 124 and made[0]["custom"]
    cid = made[0]["id"]
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before)
    open_editor(page, stem)
    pick_class(page, cid)
    page.keyboard.press("r")
    click_pixel(page, x, y)
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    assert (mask_array(stem) == cid).any()
    api(f"/api/frames/{stem}/revert", "POST")


def test_split_class(page):
    page.goto(f"{BASE}/#taxonomy")
    page.wait_for_selector(tid("class-row"))
    page.click(f'{tid("class-row")}[data-id="{SNOW}"]')
    page.click(tid("btn-split"))
    kids = page.locator(tid("dlg-child-name"))
    kids.nth(0).fill("e2e snow a")
    kids.nth(1).fill("e2e snow b")
    page.click(tid("dlg-ok"))
    page.wait_for_function("() => document.querySelectorAll('[data-testid=\"class-row\"][data-custom=\"true\"]').length >= 2")
    cls = api("/api/taxonomy")["classes"]
    kids = [c for c in cls if c["name"] in ("e2e snow a", "e2e snow b")]
    assert len(kids) == 2 and all(k["parent"] == SNOW and k["custom"] for k in kids)
