"""Opt-in real-browser regression (AppTest cannot inspect a canvas).

Start theme_gallery.py, install playwright, then set THEME_GALLERY_URL to its
localhost URL and run this file. Uses installed Edge, no browser download.
"""
import os
from urllib.parse import urlparse

import pytest

URL = os.environ.get("THEME_GALLERY_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Set THEME_GALLERY_URL for browser acceptance")


def test_native_theme_tables_forms_and_charts(tmp_path):
    if not URL:
        pytest.skip("No gallery URL")
    assert urlparse(URL).hostname in {"localhost", "127.0.0.1"}
    api = pytest.importorskip("playwright.sync_api")
    from PIL import Image
    from io import BytesIO

    with api.sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1200}, color_scheme="light")
        page = context.new_page()
        page.goto(URL, wait_until="domcontentloaded")
        switch = page.get_by_role("switch", name="Dark mode", exact=True)
        api.expect(switch).to_be_visible(timeout=30000)
        draft = page.get_by_role("textbox", name="Unsaved note", exact=True)
        draft.fill("Keep this unsaved draft")
        canvas = page.locator('[data-testid="stDataFrame"] canvas').first

        for dark in (False, True, False, True):
            if (switch.get_attribute("aria-checked") == "true") != dark:
                switch.click()
            mode = "dark" if dark else "light"
            api.expect(switch).to_have_attribute("aria-checked", str(dark).lower())
            api.expect(page.get_by_text(f"Server theme: {mode}", exact=False)).to_be_visible()
            api.expect(draft).to_have_value("Keep this unsaved draft")
            # Screenshot the *painted* grid, not its CSS background.
            page.wait_for_timeout(250)
            png = canvas.screenshot()
            pixel = Image.open(BytesIO(png)).convert("RGB").getpixel((12, 50))
            assert (sum(pixel) / 3 < 80) if dark else (sum(pixel) / 3 > 200), (mode, pixel)
            page.screenshot(path=str(tmp_path / f"controls-{mode}.png"), full_page=True)

        page.locator('[data-testid="stDataFrame"] .dvn-scroller').first.dblclick(position={"x": 110, "y": 52})
        page.keyboard.press("Control+a")
        page.keyboard.insert_text("Edited goal")
        page.keyboard.press("Enter")
        api.expect(page.get_by_text("First goal: Edited goal", exact=True)).to_be_visible()
        switch.click()
        api.expect(page.get_by_text("First goal: Edited goal", exact=True)).to_be_visible()

        # A native menu change must synchronize the sidebar and Python charts.
        page.get_by_role("button", name="Main menu", exact=True).click()
        page.get_by_test_id("stMainMenuItem-theme-Dark").click()
        api.expect(switch).to_have_attribute("aria-checked", "true")
        api.expect(page.get_by_text("Server theme: dark", exact=False)).to_be_visible()
        page.get_by_role("button", name="Main menu", exact=True).click()
        page.get_by_role("radio", name="Charts", exact=True).click()
        api.expect(page.locator('[data-testid="stPlotlyChart"]')).to_have_count(2)
        api.expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        for mode in ("dark", "light"):
            if mode == "light":
                switch.click()
            api.expect(page.get_by_text(f"Server theme: {mode}", exact=False)).to_be_visible()
            # Recoloring a line must not introduce a spurious color scale.
            api.expect(page.locator('[data-testid="stPlotlyChart"]').first.locator('.colorbar')).to_have_count(0)
            page.screenshot(path=str(tmp_path / f"charts-{mode}.png"), full_page=True)
        page.reload(wait_until="domcontentloaded")
        api.expect(switch).to_have_attribute("aria-checked", "false")
        api.expect(page.get_by_text("Server theme: light", exact=False)).to_be_visible()
        context.close()
        browser.close()
    print(f"Browser screenshots: {tmp_path}")
