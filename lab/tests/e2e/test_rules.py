from conftest import BASE, SNOW, api, tid


def test_rules_dry_run_snow_to_water(page, classes):
    water = classes["nature--water"]["id"]
    via_api = api("/api/rules/dry-run", "POST", {"from": SNOW, "to": water, "filter": {}})
    assert via_api["frames"] >= 3
    page.goto(f"{BASE}/#rules")
    page.wait_for_selector(tid("rule-from"))
    page.select_option(tid("rule-from"), str(SNOW))
    page.select_option(tid("rule-to"), str(water))
    page.click(tid("btn-dry-run"))
    page.wait_for_selector(tid("rule-result"))
    assert int(page.locator(tid("rule-result")).get_attribute("data-frames")) >= 3
    assert int(page.locator(tid("rule-result")).get_attribute("data-frames")) == via_api["frames"]
