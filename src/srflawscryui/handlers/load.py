from pathlib import Path

import gradio as gr
import pandas as pd

from srflawscry import (
    MetricRunner,
    default_config,
)
from srflawscryui.utils.metrics import (
    active_metric_names,
    fancy_diff,
    format_metric_value,
)

LOAD_PROGRESS = gr.Progress()


def _progress_callback(progress: gr.Progress):
    return lambda value, desc: progress(value, desc=desc)


def _active_metrics(metrics_df: pd.DataFrame, csv_metrics: set[str]) -> list[str]:
    active = active_metric_names(csv_metrics)
    if not active:
        raise gr.Error("No valid metrics found.")

    active = [
        metric
        for metric in active
        if not metrics_df[[f"{metric}_a", f"{metric}_b"]].isna().all().any()
    ]
    if not active:
        raise gr.Error("No comparable metrics found for matching items.")

    return active


def _per_item_table(metrics_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    per_item_table = pd.DataFrame({"item_id": metrics_df["item_id"]})
    for metric in metrics:
        per_item_table[metric] = (
            metrics_df[f"{metric}_b"] - metrics_df[f"{metric}_a"]
        ).apply(lambda value, metric=metric: fancy_diff(value, metric))
    return per_item_table.reset_index(drop=True)


def comparison_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df, active = merged_metrics_df(df_a, df_b)

    first_metric = active[0]
    metrics_df = (
        metrics_df.assign(
            _first_metric_diff=(
                metrics_df[f"{first_metric}_b"] - metrics_df[f"{first_metric}_a"]
            ),
        )
        .sort_values(
            "_first_metric_diff",
            ascending=False,
            kind="mergesort",
        )
        .drop(columns="_first_metric_diff")
        .reset_index(drop=True)
    )

    rows = [
        {
            "Metric": metric,
            name_a: format_metric_value(metrics_df[f"{metric}_a"].mean()),
            name_b: format_metric_value(metrics_df[f"{metric}_b"].mean()),
            "Diff": fancy_diff(
                metrics_df[f"{metric}_b"].mean() - metrics_df[f"{metric}_a"].mean(),
                metric,
                include_sort_key=False,
            ),
        }
        for metric in active
    ]
    average_table = pd.DataFrame(rows)

    return average_table, _per_item_table(metrics_df, active)


def merged_metrics_df(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    csv_metrics = set(df_a.columns) & set(df_b.columns) - {"item_id"}
    metrics_df = pd.merge(df_a, df_b, on="item_id", suffixes=("_a", "_b"))
    if metrics_df.empty:
        raise gr.Error("No matching item_id found.")

    active = _active_metrics(metrics_df, csv_metrics)
    return metrics_df.reset_index(drop=True), active


def saved_metrics_comparison(
    dir_a: str,
    dir_b: str,
):
    config = default_config()
    layout_a = config.layout(Path(dir_a))
    layout_b = config.layout(Path(dir_b))
    df_a = pd.read_csv(layout_a.metrics_path, dtype={"item_id": str})
    df_b = pd.read_csv(layout_b.metrics_path, dtype={"item_id": str})
    metrics_df, active = merged_metrics_df(df_a, df_b)
    return layout_a, layout_b, metrics_df, active


def sort_per_item_table_handler(
    sort_request: str,
    dir_a: str,
    dir_b: str,
) -> pd.DataFrame:
    if not dir_a or not dir_b:
        raise gr.Error("Run analysis before sorting the item list.")

    import json

    try:
        request = json.loads(sort_request or "{}")
    except json.JSONDecodeError as error:
        raise gr.Error("Invalid sort request.") from error

    metric = str(request.get("metric", "")).replace("⋮", "").strip()
    ascending = bool(request.get("ascending", False))
    _, _, metrics_df, active = saved_metrics_comparison(dir_a, dir_b)
    if metric in active:
        sort_key = metrics_df[f"{metric}_b"] - metrics_df[f"{metric}_a"]
    else:
        raise gr.Error(f"Cannot sort by unknown metric: {metric}")

    metrics_df = (
        metrics_df.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=ascending, kind="mergesort")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    return _per_item_table(metrics_df, active)


def compute_and_load_handler(
    path_a: str,
    path_b: str,
    progress: gr.Progress = LOAD_PROGRESS,
):
    metrics = ()
    try:
        if not path_a or not path_b:
            raise gr.Error("Both result directories are required.")

        config = default_config()
        metrics = config.metrics
        roots = [Path(path_a), Path(path_b)]
        for root in roots:
            if not root.is_dir():
                raise gr.Error(f"Directory not found: {root}")

        layouts = [config.layout(root) for root in roots]
        runner = MetricRunner(config.metrics)
        dataframes = runner.evaluate_many(
            layouts,
            progress=_progress_callback(progress),
        )
    except gr.Error:
        for metric in metrics:
            metric.close()
        progress(None)
        raise
    except Exception as error:
        for metric in metrics:
            metric.close()
        progress(None)
        raise gr.Error(str(error)) from error

    df_a, df_b = dataframes
    name_a, name_b = layouts[0].name, layouts[1].name

    average_table, per_item_table = comparison_tables(
        df_a,
        df_b,
        name_a,
        name_b,
    )

    progress(None)
    return (
        average_table,
        per_item_table,
        str(roots[0]),
        str(roots[1]),
    )
