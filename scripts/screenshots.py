"""Capture README / showcase screenshots of the running app.

    streamlit run app.py --server.port 8765 --server.headless true     # in one terminal
    python scripts/screenshots.py                                       # in another

Uses Playwright with the locally installed Chrome (no browser download).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screens"
TABS = {"order-list": "Order list", "outlook": "7-day outlook", "average-needs": "Average needs",
        "accuracy": "Model accuracy", "impact": "Business impact"}


def settle(page, ms=2500):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2,
                                      color_scheme=scheme)
            page = ctx.new_page()
            page.goto(URL)
            page.get_by_text("Pizza Ingredient Forecaster").first.wait_for(timeout=60000)
            settle(page, 4000)
            for slug, label in TABS.items():
                page.get_by_role("tab", name=label).click()
                settle(page)
                page.screenshot(path=OUT / f"{slug}-{scheme}.jpg", type="jpeg", quality=85)
                print("saved", f"{slug}-{scheme}.jpg")
            # Close-up crops for the showcase page (main area only, no sidebar)
            if scheme == "dark":
                page.get_by_role("tab", name="Order list").click()
                settle(page)
                main_area = page.locator('[data-testid="stMain"]')
                main_area.screenshot(path=OUT / "main-order-list-dark.jpg", type="jpeg", quality=85)
                print("saved main-order-list-dark.jpg")
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
