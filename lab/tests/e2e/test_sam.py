import pytest
from conftest import api, click_pixel, find_pixel, mask_array, open_editor, pick_class, state, tid


def test_sam_click_apply(page, clean_frame, sam_available, snow_stems):
    if not sam_available:
        pytest.skip("SAM service unavailable (/api/meta sam.available is false)")
    stem = clean_frame()
    before = mask_array(stem)
    x, y, old = find_pixel(before, not_class=0)
    open_editor(page, stem)
    pick_class(page, 62)
    page.keyboard.press("s")
    click_pixel(page, x, y)
    page.wait_for_function("() => window.__lab.state().samCandidates > 0", timeout=60_000)
    assert 1 <= state(page)["samCandidates"] <= 3
    page.keyboard.press("Enter")
    page.keyboard.press("Control+s")
    page.wait_for_function("() => !window.__lab.state().dirty")
    after = mask_array(stem)
    changed = after != before
    assert changed.sum() > 100
    assert changed[y, x]
    assert (after[changed] == 62).all()
