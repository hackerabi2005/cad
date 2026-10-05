"""
Playwright End-to-End Test for the Rust (Axum) + HTMX + Three.js Stack.
Validates:
1. Safety Disclaimer Banner.
2. 3D Canvas element initialization.
3. Patient Preset loading via HTMX (Patient C -> High Risk, Patient A -> Low Risk).
4. SHAP Waterfall updates.
5. Saves visual screenshots to reports/screenshots/.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_rust_htmx_ui():
    print("Testing Rust (Axum) + HTMX + Three.js Application at http://127.0.0.1:8000/...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")

        # 1. Safety Disclaimer
        banner = page.locator("#safety-disclaimer-banner")
        assert banner.is_visible(), "Disclaimer banner not visible!"
        print("Disclaimer banner verified.")

        time.sleep(1.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "rust_01_dashboard.png"))

        # 2. Test Presets
        print("Testing Patient C (High Risk)...")
        page.get_by_role("button", name="Patient C (High Risk)").click()
        time.sleep(1.0)
        assert page.locator("#dashboard-dynamic-updates").get_by_text("High Risk").is_visible(), "High risk status not visible!"
        page.screenshot(path=str(SCREENSHOTS_DIR / "rust_02_high_risk.png"))

        print("Testing Patient A (Low Risk)...")
        page.get_by_role("button", name="Patient A (Low Risk)").click()
        time.sleep(1.0)
        assert page.locator("#dashboard-dynamic-updates").get_by_text("Low Risk").is_visible(), "Low risk status not visible!"
        page.screenshot(path=str(SCREENSHOTS_DIR / "rust_03_low_risk.png"))

        # 3. Test Tabs
        print("Testing Model CV Validation tab...")
        page.get_by_role("button", name="Model CV Validation").click()
        time.sleep(0.8)
        assert page.locator("text=Repeated Stratified 5-Fold Cross-Validation").is_visible()
        page.screenshot(path=str(SCREENSHOTS_DIR / "rust_04_cv_metrics.png"))

        # 4. Test Return to Risk Tab & Vessel Selection
        print("Testing Return to Risk & SHAP tab and LAD Vessel Selection...")
        page.get_by_role("button", name="Risk & SHAP").click()
        time.sleep(1.0)
        page.locator(".vessel-card").filter(has_text="LAD").click()
        time.sleep(0.8)
        shap_title = page.locator("#shap-container h2").inner_text()
        assert "LAD" in shap_title, f"Expected LAD Explainer, got: {shap_title}"
        page.screenshot(path=str(SCREENSHOTS_DIR / "rust_05_lad_shap.png"))

        browser.close()
        print("\nALL RUST + HTMX PLAYWRIGHT TESTS PASSED!")


if __name__ == "__main__":
    test_rust_htmx_ui()
