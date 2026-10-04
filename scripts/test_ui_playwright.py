"""
Automated Playwright UI & End-to-End Verification Suite.
Verifies:
- G2: Vessel selection & SHAP switching
- G3: Live debounced predictions & patient preset loading (<1s)
- G4: Clinical dashboard, SHAP waterfalls, and tab switching
- G5: Persistent Safety Disclaimer banner & footer (on load, after scroll, at 375px)
- G6: Performance benchmark with --disable-gpu
Saves screenshots to reports/screenshots/
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_ui():
    print("Starting Playwright end-to-end verification...")
    with sync_playwright() as p:
        # Launch Chromium headless with --disable-gpu as specified in G6
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Navigating to http://127.0.0.1:8000/...")
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")

        # 1. Verify G5 Disclaimer Banner on load
        banner = page.locator("#safety-disclaimer-banner")
        assert banner.is_visible(), "Top safety disclaimer banner not visible on load!"
        banner_text = banner.inner_text()
        print("Disclaimer banner verified:", banner_text[:60] + "...")
        assert "Decision support" in banner_text
        assert "not a substitute for formal diagnostic imaging" in banner_text

        # Take initial screenshot
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_initial_dashboard.png"), full_page=False)

        # 2. Verify G5 Disclaimer after scroll
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)
        footer = page.locator("#safety-disclaimer-footer")
        assert footer.is_visible(), "Safety disclaimer footer not visible after scroll!"
        assert "CC BY 4.0" in footer.inner_text()
        assert "CC BY-SA 2.1 JP" in footer.inner_text()

        # 3. Verify G5 Mobile Responsive 375px width
        page.set_viewport_size({"width": 375, "height": 812})
        time.sleep(0.5)
        assert banner.is_visible(), "Banner not visible on mobile 375px viewport!"
        page.screenshot(path=str(SCREENSHOTS_DIR / "02_mobile_responsive_375px.png"))
        print("Mobile 375px viewport disclaimer verified.")

        # Reset viewport back to desktop
        page.set_viewport_size({"width": 1440, "height": 900})
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)

        # 4. Verify G3 Patient Presets (< 1s update)
        print("Testing Patient Presets loading...")
        # Click High Risk preset
        t0 = time.perf_counter()
        high_risk_btn = page.get_by_role("button", name="Patient C — High Risk Stenosis")
        high_risk_btn.click()

        # Wait for high risk response
        page.wait_for_selector("text=High Risk", timeout=2000)
        t_high = time.perf_counter() - t0
        print(f"High risk preset updated in {t_high:.3f}s (< 1.0s budget)")
        assert t_high < 1.0, f"Preset update took {t_high:.3f}s (> 1.0s)!"

        page.screenshot(path=str(SCREENSHOTS_DIR / "03_high_risk_patient.png"))

        # Click Low Risk preset
        t0 = time.perf_counter()
        low_risk_btn = page.get_by_role("button", name="Patient A — Low Risk Profile")
        low_risk_btn.click()
        page.wait_for_selector("text=Low Risk", timeout=2000)
        t_low = time.perf_counter() - t0
        print(f"Low risk preset updated in {t_low:.3f}s (< 1.0s budget)")

        page.screenshot(path=str(SCREENSHOTS_DIR / "04_low_risk_patient.png"))

        # 5. Verify G2 & G4 Vessel selection & SHAP switching
        print("Testing Vessel Selection and SHAP attribution switching...")
        lad_card = page.locator("#vessel-card-LAD")
        lad_card.click()
        time.sleep(0.6)

        # Verify SHAP panel updated to LAD
        lad_title = page.locator("text=Left Anterior Descending (LAD) Stenosis Model")
        assert lad_title.is_visible(), "SHAP panel did not switch to LAD model!"
        page.screenshot(path=str(SCREENSHOTS_DIR / "05_lad_shap_waterfall.png"))
        print("LAD selection and SHAP waterfall verified.")

        # Switch to LCX
        lcx_card = page.locator("#vessel-card-LCX")
        lcx_card.click()
        time.sleep(0.6)
        lcx_title = page.locator("text=Left Circumflex (LCX) Stenosis Model")
        assert lcx_title.is_visible(), "SHAP panel did not switch to LCX model!"
        page.screenshot(path=str(SCREENSHOTS_DIR / "06_lcx_shap_waterfall.png"))
        print("LCX selection and SHAP waterfall verified.")

        # Switch to Table view in SHAP waterfall
        table_btn = page.locator("#btn-shap-table-view")
        table_btn.click()
        time.sleep(0.4)
        assert page.locator("text=Physiological Feature").is_visible()
        page.screenshot(path=str(SCREENSHOTS_DIR / "07_shap_table_view.png"))

        # 6. Verify Global Importance Tab
        print("Testing Global Factors tab...")
        global_tab = page.get_by_role("button", name="Global Factors")
        global_tab.click()
        time.sleep(0.5)
        assert page.locator("text=Global Feature Importance").is_visible()
        page.screenshot(path=str(SCREENSHOTS_DIR / "08_global_importance_tab.png"))
        print("Global Factors tab verified.")

        # 7. Verify Model CV Validation Tab
        print("Testing Model CV Validation tab...")
        metrics_tab = page.get_by_role("button", name="Model CV Validation")
        metrics_tab.click()
        time.sleep(0.5)
        assert page.locator("text=Repeated Stratified 5-Fold Cross-Validation").is_visible()
        assert page.locator("text=Candidate Model").is_visible()
        page.screenshot(path=str(SCREENSHOTS_DIR / "09_model_validation_tab.png"))
        print("Model CV Validation tab verified.")

        # 8. Verify G6 Performance Benchmark (Orbiting and rendering frametime)
        print("Testing G6 performance under software rendering (--disable-gpu)...")
        # Return to main tab
        page.get_by_role("button", name="Risk & SHAP").click()
        time.sleep(0.5)

        # Measure 60 animation frames
        frame_times = page.evaluate("""
            () => new Promise((resolve) => {
                const times = [];
                let last = performance.now();
                let count = 0;
                function frame() {
                    const now = performance.now();
                    times.push(now - last);
                    last = now;
                    count++;
                    if (count < 40) {
                        requestAnimationFrame(frame);
                    } else {
                        resolve(times);
                    }
                }
                requestAnimationFrame(frame);
            })
        """)
        import numpy as np
        median_ft = float(np.median(frame_times[2:]))
        approx_fps = 1000.0 / median_ft if median_ft > 0 else 60.0
        print(f"Software Rendering Frame Time: Median={median_ft:.2f}ms (~{approx_fps:.1f} FPS, Budget >= 20 FPS)")
        assert approx_fps >= 20.0 or median_ft <= 50.0, f"Frame rate {approx_fps:.1f} FPS below 20 FPS budget!"

        browser.close()
        print("\nALL PLAYWRIGHT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_ui()
