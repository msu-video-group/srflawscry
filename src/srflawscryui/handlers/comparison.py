import gradio as gr
import numpy as np
import pandas as pd

from srflawscry import AbstractEvaluationLayout
from srflawscryui.handlers.load import saved_metrics_comparison
from srflawscryui.utils.contours import draw_contours
from srflawscryui.utils.metrics import (
    fancy_diff,
    format_metric_value,
    mask_paths_for_row,
)


def _row_index(selected: gr.SelectData) -> int:
    if isinstance(selected.index, int):
        return selected.index
    return selected.index[0]


def side_by_side_handler(
    selected_row: gr.SelectData,
    per_item_table: pd.DataFrame,
    dir_a: str,
    dir_b: str,
):
    if per_item_table is None or dir_a is None or dir_b is None:
        raise gr.Error("Run analysis before opening side-by-side comparison.")

    item_id = per_item_table.iloc[_row_index(selected_row)]["item_id"]
    layout_a, layout_b, metrics_df, metric_names = saved_metrics_comparison(
        dir_a,
        dir_b,
    )
    row = metrics_df.loc[metrics_df["item_id"] == item_id].iloc[0]

    def draw_contours_from(layout: AbstractEvaluationLayout, suffix: str) -> np.ndarray:
        sample = layout.sample(item_id)
        mask_paths = mask_paths_for_row(layout, row, suffix, metric_names)
        return draw_contours(
            sample.result_path,
            mask_paths,
        )

    out_a = draw_contours_from(layout_a, "a")
    out_b = draw_contours_from(layout_b, "b")

    rows = [
        {
            "Metric": metric,
            layout_a.name: format_metric_value(row[f"{metric}_a"]),
            layout_b.name: format_metric_value(row[f"{metric}_b"]),
            "Diff": fancy_diff(row[f"{metric}_b"] - row[f"{metric}_a"], metric),
        }
        for metric in metric_names
    ]
    item_table = pd.DataFrame(rows)

    return (
        gr.update(value=item_table, label=f"Item: {item_id}"),
        gr.update(value=out_a, label=layout_a.name),
        gr.update(value=out_b, label=layout_b.name),
    )
