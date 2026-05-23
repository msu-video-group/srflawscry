from collections.abc import Callable
from typing import Any

import gradio as gr

from srflawscryui.demo_data import demo_result_dirs
from srflawscryui.handlers.comparison import side_by_side_handler
from srflawscryui.handlers.load import (
    compute_and_load_handler,
    sort_per_item_table_handler,
)

APP_CSS = """
#setup-actions {
  align-items: end;
}

#run-analysis-button {
  min-width: 180px;
}

#analysis-progress-slot {
  min-height: 48px;
}

.gradio-container .progress-bar,
.gradio-container .progress-text {
  margin-top: 0.5rem;
}

.srscry-tab-disabled {
  cursor: not-allowed !important;
  opacity: 0.45;
}

#per-item-table th,
#per-item-table [role="columnheader"] {
  cursor: pointer;
}
"""

INSTALL_TAB_GUARD_JS = """
() => {
  const state = window.__srscryTabs ??= {
    comparisonReady: false,
    sideReady: false,
    installed: false,
  };
  const tabByName = (name) => Array.from(document.querySelectorAll('[role="tab"]'))
    .find((element) => element.textContent.trim() === name);
  const setBlocked = (name, blocked) => {
    const tab = tabByName(name);
    if (!tab) return;
    tab.setAttribute("aria-disabled", blocked ? "true" : "false");
    tab.classList.toggle("srscry-tab-disabled", blocked);
  };
  window.__srscryRefreshTabs = () => {
    setBlocked("Comparison", !state.comparisonReady);
    setBlocked("Side-by-Side", !state.sideReady);
  };
  if (!state.installed) {
    document.addEventListener("click", (event) => {
      const tab = event.target.closest?.('[role="tab"]');
      if (!tab) return;
      const name = tab.textContent.trim();
      const blocked = (
        (name === "Comparison" && !state.comparisonReady) ||
        (name === "Side-by-Side" && !state.sideReady)
      );
      if (!blocked) return;
      event.preventDefault();
      event.stopPropagation();
      window.__srscryRefreshTabs();
    }, true);
    state.installed = true;
  }
  requestAnimationFrame(window.__srscryRefreshTabs);
  setTimeout(window.__srscryRefreshTabs, 500);
}
"""

OPEN_COMPARISON_TAB_JS = """
() => {
  const state = window.__srscryTabs ??= {};
  state.comparisonReady = true;
  state.sideReady = false;
  if (window.__srscryPerItemSort) {
    window.__srscryPerItemSort.metric = null;
    window.__srscryPerItemSort.direction = null;
  }
  window.__srscryRefreshTabs?.();
  requestAnimationFrame(() => {
    const tab = Array.from(document.querySelectorAll('[role="tab"]'))
      .find((element) => element.textContent.trim() === "Comparison");
    window.__srscryRefreshTabs?.();
    tab?.click();
  });
}
"""

OPEN_SIDE_BY_SIDE_TAB_JS = """
() => {
  const state = window.__srscryTabs ??= {};
  state.comparisonReady = true;
  state.sideReady = true;
  window.__srscryRefreshTabs?.();
  requestAnimationFrame(() => {
    const tab = Array.from(document.querySelectorAll('[role="tab"]'))
      .find((element) => element.textContent.trim() === "Side-by-Side");
    window.__srscryRefreshTabs?.();
    tab?.click();
  });
}
"""

SHOW_COMPARISON_TAB_JS = """
() => {
  window.__srscryRefreshTabs?.();
  requestAnimationFrame(() => {
    const tab = Array.from(document.querySelectorAll('[role="tab"]'))
      .find((element) => element.textContent.trim() === "Comparison");
    window.__srscryRefreshTabs?.();
    tab?.click();
  });
}
"""

INSTALL_PER_ITEM_SORT_JS = """
() => {
  const state = window.__srscryPerItemSort ??= {
    installed: false,
    metric: null,
    direction: null,
  };
  const sortRequestInput = () => document.querySelector(
    '#per-item-sort-request textarea, #per-item-sort-request input'
  );
  const headerText = (header) => header.textContent
    .replace(/[▲▼⋮]/g, "")
    .trim();

  if (state.installed) return;

  document.addEventListener("click", (event) => {
    const root = document.querySelector("#per-item-table");
    if (!root || !root.contains(event.target)) return;

    const header = event.target.closest?.('th, [role="columnheader"]');
    if (!header || !root.contains(header)) return;

    const metric = headerText(header);
    if (!metric || metric === "item_id") return;

    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();

    const direction = (
      state.metric === metric && state.direction === "desc" ? "asc" : "desc"
    );
    state.metric = metric;
    state.direction = direction;

    const input = sortRequestInput();
    if (!input) return;

    input.value = JSON.stringify({
      metric,
      ascending: direction === "asc",
      nonce: Date.now(),
    });
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, true);

  state.installed = true;
}
"""

LAUNCH_KWARGS = {"ssr_mode": False}


def build_app(
    compute_handler: Callable[..., tuple[Any, ...]] = compute_and_load_handler,
) -> gr.Blocks:
    default_dir_a, default_dir_b = demo_result_dirs()

    with gr.Blocks(title="SR Flaw Scry UI") as app:
        gr.HTML(f"<style>{APP_CSS}</style>")
        with gr.Tabs(selected="setup_view"):
            with gr.Tab("Setup", id="setup_view"):
                gr.Markdown("### Setup comparison")
                with gr.Row():
                    dir_a_textbox = gr.Textbox(
                        label="Path to SR A results",
                        value=default_dir_a,
                    )
                    dir_b_textbox = gr.Textbox(
                        label="Path to SR B results",
                        value=default_dir_b,
                    )
                with gr.Row(elem_id="setup-actions"):
                    load_btn = gr.Button(
                        "🚀 Run Analysis",
                        variant="primary",
                        elem_id="run-analysis-button",
                    )
                progress_anchor = gr.HTML("", elem_id="analysis-progress-slot")

            with gr.Tab("Comparison", id="comparison_view"):
                gr.Markdown("### Global Comparison")
                average_table = gr.Dataframe(
                    interactive=False, label="Average Metrics", datatype="html"
                )
                gr.Markdown("---")
                gr.Markdown("### Per-Item Comparison")
                per_item_table = gr.Dataframe(
                    interactive=False,
                    label="Item List (Click a row to inspect)",
                    datatype="html",
                    elem_id="per-item-table",
                )

            with gr.Tab("Side-by-Side", id="sbs_view"):
                item_table = gr.Dataframe(
                    interactive=False,
                    label="Item: None, Mask: None",
                    datatype="html",
                )
                with gr.Row():
                    image_a = gr.Image(label="SR A", type="pil")
                    image_b = gr.Image(label="SR B", type="pil")
                back_btn = gr.Button("⇽ Back", size="lg")

        dir_a_path = gr.State()
        dir_b_path = gr.State()
        sort_request = gr.Textbox(
            value="",
            visible="hidden",
            elem_id="per-item-sort-request",
        )

        app.load(fn=None, js=INSTALL_TAB_GUARD_JS)
        app.load(fn=None, js=INSTALL_PER_ITEM_SORT_JS)

        load_event = load_btn.click(
            fn=compute_handler,
            inputs=[
                dir_a_textbox,
                dir_b_textbox,
            ],
            outputs=[
                average_table,
                per_item_table,
                dir_a_path,
                dir_b_path,
            ],
            show_progress="full",
            show_progress_on=progress_anchor,
        )
        load_event.then(fn=None, js=OPEN_COMPARISON_TAB_JS)

        sort_request.change(
            fn=sort_per_item_table_handler,
            inputs=[sort_request, dir_a_path, dir_b_path],
            outputs=[per_item_table],
            show_progress="hidden",
            queue=False,
        )

        inspect_event = per_item_table.select(
            fn=side_by_side_handler,
            inputs=[
                per_item_table,
                dir_a_path,
                dir_b_path,
            ],
            outputs=[item_table, image_a, image_b],
        )
        inspect_event.then(fn=None, js=OPEN_SIDE_BY_SIDE_TAB_JS)

        back_btn.click(fn=None, js=SHOW_COMPARISON_TAB_JS)

    return app


def main(argv: list[str] | None = None):
    launch_app(argv)


def launch_app(argv: list[str] | None = None):
    _ = argv
    app = build_app()
    app.launch(**LAUNCH_KWARGS)


if __name__ == "__main__":
    main()
