"""Inspect tool (default): click selects the 4-connected region of the class under the pointer; Esc clears; Enter relabels."""
import cv2
import numpy as np
from conftest import H, W, SNOW, api, find_pixel, click_pixel, mask_array, open_editor, pick_class, pixel, state, tid, to_client, zoom_to


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
    assert sel["edgeLen"] == edge_len(comp) and sel["contours"] == n_contours(comp)
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


def edge_len(comp):
    """Number of 4-neighbour boundary edges of a region (the image border counts as outside)."""
    p = np.pad(comp, 1)
    return int((p[1:, :] != p[:-1, :]).sum() + (p[:, 1:] != p[:, :-1]).sum())


def n_contours(comp):
    """Closed boundary loops = outer boundary + holes = 8-connected components of the padded complement."""
    n, _ = cv2.connectedComponents((~np.pad(comp, 1)).astype(np.uint8), connectivity=8)
    return n - 1


def find_holed_region(frames, min_px=3000):
    """(stem, x, y, comp) of a region with a hole whose seed pixel has a uniform 9x9 neighbourhood."""
    for r in frames:
        arr = mask_array(r["stem"])
        for y in range(20, H - 20, 12):
            for x in range(20, W - 20, 12):
                v = arr[y - 4:y + 5, x - 4:x + 5]
                if not (v == v[4, 4]).all():
                    continue
                comp, _ = region_of(arr, x, y)
                if comp.sum() >= min_px and n_contours(comp) >= 2:
                    return r["stem"], x, y, comp
    return None


def sample(page, cx, cy, r=2):
    """RGB samples of the display canvas around a client point (r pixel radius), as a list of [r, g, b]."""
    return page.evaluate("""([cx, cy, r]) => { const c = document.querySelector('[data-testid="editor-canvas"]'), b = c.getBoundingClientRect(), d = c.width / b.width;
        const x = Math.round((cx - b.left) * d) - r, y = Math.round((cy - b.top) * d) - r, n = 2 * r + 1;
        const px = c.getContext('2d').getImageData(x, y, n, n).data, out = [];
        for (let i = 0; i < px.length; i += 4) out.push([px[i], px[i + 1], px[i + 2]]);
        return out; }""", [cx, cy, r])


def has_color(samples, color, tol=6):
    return any(all(abs(a - b) <= tol for a, b in zip(s, color)) for s in samples)


def test_selection_is_outlined_by_its_boundary_with_holes(page, frames):
    found = find_holed_region(frames)
    assert found, "no region with a hole in the fixture"
    stem, x, y, comp = found
    arr = mask_array(stem)
    cls = int(arr[y, x])
    color = next(c["color"] for c in api("/api/taxonomy")["classes"] if c["id"] == cls)
    open_editor(page, stem)
    click_pixel(page, x, y)                                             # fit zoom: the 9x9 uniform seed makes the click safe
    sel = state(page)["selection"]
    assert sel["id"] == cls and sel["px"] == int(comp.sum())
    assert sel["contours"] == n_contours(comp) >= 2                     # outer boundary + at least one hole
    assert sel["edgeLen"] == edge_len(comp)
    # a boundary pixel: left neighbour outside the region -> the vertical edge at image x; the class-colour stroke sits on it
    ys, xs = np.nonzero(comp & ~np.pad(comp, ((0, 0), (1, 0)))[:, :-1] & (np.arange(W)[None, :] > 3))
    k = len(xs) // 2
    bx, by = int(xs[k]), int(ys[k])
    c = to_client(page, bx - 0.5, by)                                   # client point of the edge at image x = bx, row centre
    assert has_color(sample(page, c["x"], c["y"], 2), color), "no class-colour stroke on the boundary"
    # deep inside the region: only the tint, no stroke
    dist = cv2.distanceTransform(np.pad(comp, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    iy, ix = np.unravel_index(int(dist.argmax()), dist.shape)
    assert dist[iy, ix] >= 6
    ci = to_client(page, int(ix), int(iy))
    assert not has_color(sample(page, ci["x"], ci["y"], 2), color, tol=3), "stroke found deep inside the region"
    # no bounding rectangle: on the bbox's top edge, away from the region, there is no stroke either
    x0, y0, x1, y1 = sel["bbox"]
    away = cv2.distanceTransform((~np.pad(comp, 1)).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    for tx in range(x0, x1, 4):
        if not comp[y0, tx] and away[y0, tx] >= 8:
            ct = to_client(page, tx, y0)
            assert not has_color(sample(page, ct["x"], ct["y"], 1), color, tol=3)
            break
