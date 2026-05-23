from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from srflawscry import AbstractEvaluationLayout, MetricSpec, default_config


@dataclass(frozen=True)
class MetricUiConfig:
    metric_id: str
    higher_is_better: bool
    tolerance: float
    mask_name: str


METRIC_SPECS: dict[str, tuple[str, MetricSpec]] = {
    spec.name: (metric.metric_id, spec)
    for metric in default_config().metrics
    for spec in metric.specs
}
METRIC_CONFIG: dict[str, MetricUiConfig] = {
    name: MetricUiConfig(
        metric_id=metric_id,
        higher_is_better=spec.higher_is_better,
        tolerance=spec.tolerance,
        mask_name=spec.mask_name,
    )
    for name, (metric_id, spec) in METRIC_SPECS.items()
}


def active_metric_names(columns: set[str]) -> list[str]:
    return [metric for metric in METRIC_CONFIG if metric in columns]


def mask_outputs_for_row(
    row,
    suffix: str,
    metric_names: list[str],
) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    for metric_name in metric_names:
        config = METRIC_CONFIG[metric_name]
        mask_name = config.mask_name
        if not mask_name:
            continue

        column = f"{metric_name}_{suffix}"
        value = float(row[column])
        active = math.isfinite(value) and abs(value) > 0.0
        output = (config.metric_id, mask_name)
        if active and output not in outputs:
            outputs.append(output)
    return outputs


def mask_paths_for_row(
    layout: AbstractEvaluationLayout,
    row,
    suffix: str,
    metric_names: list[str],
) -> list[Path]:
    paths: list[Path] = []
    for metric_id, mask_name in mask_outputs_for_row(row, suffix, metric_names):
        paths.append(
            layout.metric_output_path(
                metric_id,
                mask_name,
                row["item_id"],
            )
        )
    return paths


def format_metric_value(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def fancy_diff(
    diff: float,
    metric_name: str,
    zero_as_plain: bool = False,
    include_sort_key: bool = True,
) -> str:
    config = METRIC_CONFIG[metric_name]
    color = "currentColor"
    diff = float(diff)
    if not math.isfinite(diff):
        return '<span style="color: currentColor; font-family: monospace;">n/a</span>'

    if abs(diff) >= config.tolerance:
        if config.higher_is_better:
            color = "#2ecc71" if diff > 0 else "#e74c3c"
        else:
            color = "#2ecc71" if diff < 0 else "#e74c3c"

    # Hidden Sort Key
    sort_key = f"{1000 + diff:09.4f}"

    sort_prefix = (
        f'<span style="display:none;">{sort_key}</span>' if include_sort_key else ""
    )
    value = "0" if zero_as_plain and abs(diff) < 0.00005 else f"{diff:+.4f}"
    return (
        sort_prefix
        + f'<span style="color: {color}; font-family: monospace;">{value}</span>'
    )
