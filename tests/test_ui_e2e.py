from __future__ import annotations

import contextlib
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from srflawscryui.app import build_app
from srflawscryui.handlers.load import comparison_tables

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright

CHROMIUM_PATH = Path("/usr/bin/chromium-browser")


def _fake_compute_handler(
    path_a,
    path_b,
    progress: gr.Progress | None = None,
):
    if progress is None:
        progress = gr.Progress()

    progress((0, 2), desc="Loading fake metric weights")
    time.sleep(0.8)
    progress((1, 2), desc="Computing WASD metric")
    time.sleep(0.8)

    df_a = pd.DataFrame(
        [
            {
                "item_id": "negative.png",
                "hallucinated_texture": 0.9,
            },
            {
                "item_id": "positive.png",
                "hallucinated_texture": 0.3,
            },
            {
                "item_id": "small.png",
                "hallucinated_texture": 0.1,
            },
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "item_id": "negative.png",
                "hallucinated_texture": 0.1,
            },
            {
                "item_id": "positive.png",
                "hallucinated_texture": 0.6,
            },
            {
                "item_id": "small.png",
                "hallucinated_texture": 0.2,
            },
        ]
    )
    Path(path_a, "metrics.csv").write_text(df_a.to_csv(index=False))
    Path(path_b, "metrics.csv").write_text(df_b.to_csv(index=False))
    average_table, per_item_table = comparison_tables(
        df_a,
        df_b,
        "A",
        "B",
    )
    progress(None)
    return (
        average_table,
        per_item_table,
        path_a,
        path_b,
    )


@pytest.mark.skipif(
    not CHROMIUM_PATH.exists(),
    reason="system Chromium is required for UI E2E tests",
)
def test_ui_run_analysis_progress_and_tabs(tmp_path):
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    for root, value in ((dir_a, 80), (dir_b, 160)):
        (root / "results").mkdir(parents=True)
        (root / "sources").mkdir()
        for item_id in ("negative.png", "positive.png", "small.png"):
            image = Image.fromarray(
                np.full((32, 32, 3), value, dtype=np.uint8),
                mode="RGB",
            )
            image.save(root / "results" / item_id)
            image.save(root / "sources" / item_id)

    app = build_app(compute_handler=_fake_compute_handler)
    _, local_url, _ = app.launch(
        prevent_thread_lock=True,
        quiet=True,
        server_name="127.0.0.1",
    )

    screenshot_path = tmp_path / "ui-failure.png"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(CHROMIUM_PATH),
                headless=True,
                args=["--no-sandbox"],
            )
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            try:
                page.goto(local_url, wait_until="domcontentloaded")
                page.get_by_role("button", name="Run Analysis").wait_for()
                page.get_by_role("textbox", name="Path to SR A results").fill(
                    str(dir_a)
                )
                page.get_by_role("textbox", name="Path to SR B results").fill(
                    str(dir_b)
                )

                expect(page.get_by_role("tab", name="Setup")).to_be_visible()
                expect(page.get_by_role("tab", name="Comparison")).to_be_visible()
                expect(page.get_by_role("tab", name="Side-by-Side")).to_be_visible()

                page.get_by_role("button", name="Run Analysis").click()
                expect(page.get_by_text("Loading fake metric weights")).to_be_visible()
                expect(page.get_by_text("Computing WASD metric")).to_be_visible()

                expect(page.get_by_role("tab", name="Comparison")).to_have_attribute(
                    "aria-selected",
                    "true",
                    timeout=10_000,
                )
                expect(page.get_by_text("Global Comparison")).to_be_visible()
                expect(page.get_by_text("Average Metrics").first).to_be_visible()
                expect(
                    page.get_by_text("Item List (Click a row to inspect)").first
                ).to_be_visible()
                expect(page.get_by_text("Per-Item Comparison").first).to_be_visible()

                per_item_root = page.locator("#per-item-table")
                per_item_root.get_by_text("hallucinated_texture").first.click()
                page.wait_for_timeout(1_000)
                per_item_text = per_item_root.inner_text()
                assert per_item_text.find("positive.png") < per_item_text.find(
                    "small.png"
                )
                assert per_item_text.find("small.png") < per_item_text.find(
                    "negative.png"
                )

                per_item_root.get_by_text("hallucinated_texture").first.click()
                page.wait_for_timeout(1_000)
                per_item_text = per_item_root.inner_text()
                assert per_item_text.find("negative.png") < per_item_text.find(
                    "small.png"
                )
                assert per_item_text.find("small.png") < per_item_text.find(
                    "positive.png"
                )

                page.get_by_role("button", name="negative.png").click()
                expect(page.get_by_role("tab", name="Side-by-Side")).to_have_attribute(
                    "aria-selected",
                    "true",
                    timeout=10_000,
                )
                expect(page.get_by_text("Item: negative.png").first).to_be_visible()
                expect(page.get_by_text("A").first).to_be_visible()
                expect(page.get_by_text("B").first).to_be_visible()
                expect(page.get_by_role("link", name="Download")).to_have_count(2)
            except Exception:
                with contextlib.suppress(Exception):
                    page.screenshot(path=screenshot_path, full_page=True, timeout=3_000)
                raise
            finally:
                browser.close()
    finally:
        app.close()
