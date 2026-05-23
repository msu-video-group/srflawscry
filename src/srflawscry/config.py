from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from srflawscry.layouts import AbstractEvaluationLayout, FolderWithSourcesLayout
from srflawscry.metrics import AbstractMetric
from srflawscry.metrics.wasd import WASDMetric


@dataclass(frozen=True)
class EvaluationConfig:
    layout: Callable[[Path], AbstractEvaluationLayout]
    metrics: tuple[AbstractMetric, ...]


def default_config() -> EvaluationConfig:
    return EvaluationConfig(
        layout=lambda root: FolderWithSourcesLayout(root),
        metrics=(WASDMetric(),),
    )
