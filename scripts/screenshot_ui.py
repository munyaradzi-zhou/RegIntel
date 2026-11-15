"""Screenshot every tab of the running UI (API on :8000, Streamlit on :8501) with the installed Chrome.

    python scripts/screenshot_ui.py [--out notes/screenshots/ui] [--question "..."] [--url https://...]

Asks one real question, so it spends one answer's worth of Azure tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = "http://localhost:8501/"


def settle(page: Page, text: str, timeout: int = 60_000) -> None:
    page.get_by_text(text, exact=False).locator("visible=true").first.wait_for(timeout=timeout)
    page.wait_for_timeout(1500)  # let charts and fonts finish drawing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="notes/screenshots/ui")
    parser.add_argument("--url", default=URL, help="the UI to capture (default: local Streamlit)")
    parser.add_argument("--question", default="Is there a conjunctivitis signal for dupilumab in FAERS?")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1900}, device_scale_factor=1)
        page.goto(args.url)
        settle(page, "Ask the guidance.", timeout=120_000)  # a cold container app takes a moment
        page.screenshot(path=out / "01-ask-empty.png", full_page=True)

        page.get_by_test_id("stChatInputTextArea").fill(args.question)
        page.keyboard.press("Enter")
        settle(page, "Sources (", timeout=180_000)
        page.get_by_text("Sources (").first.click()
        page.wait_for_timeout(800)
        page.screenshot(path=out / "02-ask-answer.png", full_page=True)

        for name, tab, marker in (
            ("03-safety-signals", "Safety signals", "Reporting odds ratio"),
            ("04-cdisc-terms", "CDISC terms", "Look up a term"),
            ("05-how-it-works", "How it works", "Retrieve, write, check"),
        ):
            page.get_by_role("tab", name=tab).click()
            settle(page, marker)
            page.screenshot(path=out / f"{name}.png", full_page=True)
        browser.close()
    print("saved to", out)


if __name__ == "__main__":
    main()
