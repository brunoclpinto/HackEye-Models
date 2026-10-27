import glob
import os

from conftest import BASE, DATA, api, tid

NAME = "e2e-1"   # map and dataset share the name; run.sh deletes taxonomies/e2e-*.yaml


def test_map_merge_expand_export(page):
    page.goto(f"{BASE}/#taxonomy")
    page.wait_for_selector(tid("class-row"))
    page.select_option(tid("map-load"), "HackEye26.1-48")
    page.wait_for_selector(tid("group-card"))
    n0 = int(page.locator(tid("preview-n")).inner_text().strip().split()[0])
    assert n0 == 48
    cards = page.locator(tid("group-card"))
    count0 = cards.count()
    # merge two groups
    cards.nth(0).locator(tid("group-select")).check()
    cards.nth(1).locator(tid("group-select")).check()
    page.click(tid("btn-merge"))
    page.fill(tid("dlg-name"), "e2e merged")
    page.click(tid("dlg-ok"))
    page.wait_for_function(f"() => document.querySelectorAll('[data-testid=\"group-card\"]').length === {count0 - 1}")
    n1 = int(page.locator(tid("preview-n")).inner_text().strip().split()[0])
    assert n1 < n0
    # expand one group back into its members
    page.locator(f'{tid("group-card")}[data-name="e2e merged"] {tid("group-expand")}').click()
    page.wait_for_function(f"() => document.querySelectorAll('[data-testid=\"group-card\"][data-name=\"e2e merged\"]').length === 0")
    n2 = int(page.locator(tid("preview-n")).inner_text().strip().split()[0])
    assert n2 > n1
    # merge again so the export differs from the stock map, save as a new dataset
    cards = page.locator(tid("group-card"))
    cards.nth(0).locator(tid("group-select")).check()
    cards.nth(1).locator(tid("group-select")).check()
    page.click(tid("btn-merge"))
    page.fill(tid("dlg-name"), "e2e merged2")
    page.click(tid("dlg-ok"))
    page.fill(tid("map-name"), NAME)
    page.click(tid("btn-save-map"))
    for _ in range(30):
        if NAME in [m["name"] for m in api("/api/maps")]:
            break
        page.wait_for_timeout(200)
    assert NAME in [m["name"] for m in api("/api/maps")]
    assert not os.path.exists(f"{DATA}/datasets/{NAME}")
    page.click(tid("btn-export"))
    page.wait_for_function("() => ['done','failed'].includes(document.querySelector('[data-testid=\"job-status\"]')?.dataset.status)",
                           timeout=300_000)
    assert page.locator(tid("job-status")).get_attribute("data-status") == "done"
    root = f"{DATA}/datasets/{NAME}"
    assert glob.glob(f"{root}/*.yaml"), os.listdir(root)
    n_img = sum(len(os.listdir(f"{root}/images/{s}")) for s in ("train", "val") if os.path.isdir(f"{root}/images/{s}"))
    n_msk = sum(len(os.listdir(f"{root}/masks/{s}")) for s in ("train", "val") if os.path.isdir(f"{root}/masks/{s}"))
    assert n_img == n_msk == 20
    jobs = api("/api/jobs?kind=export")
    assert jobs and jobs[0]["status"] == "done"
    assert not page.errors
