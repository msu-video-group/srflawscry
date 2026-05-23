from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from srflawscry.layouts import AbstractEvaluationLayout
from srflawscry.samples import Sample


@dataclass(frozen=True)
class MetricSpec:
    """One output column: name, direction, tolerance, and optional mask."""

    name: str
    higher_is_better: bool
    tolerance: float = 0.0
    mask_name: str = ""


@dataclass(frozen=True)
class MetricResult:
    """Computed values keyed by MetricSpec.name."""

    values: Mapping[str, float]


class AbstractMetric(ABC):
    """Metric contract; implement specs and evaluate()."""

    metric_id: str

    @property
    @abstractmethod
    def specs(self) -> tuple[MetricSpec, ...]:
        """Return output columns and UI comparison rules."""
        raise NotImplementedError

    def open(self) -> None:
        """Allocate batch resources, if the metric needs them."""
        return None

    @property
    def is_open(self) -> bool:
        """Return whether reusable resources are already allocated."""
        return False

    @property
    def open_steps(self) -> tuple[str, ...]:
        """Return named resource-loading steps shown by MetricRunner."""
        return ("Loading resources",)

    def open_step(self, step: str) -> None:
        """Allocate one named resource-loading step."""
        if step not in self.open_steps:
            raise ValueError(f"Unknown open step for {self.metric_id}: {step}")
        self.open()

    def close(self) -> None:
        """Release resources allocated by open()."""
        return None

    def has_cached_outputs(
        self,
        layout: AbstractEvaluationLayout,
        item_id: str,
    ) -> bool:
        """Return whether cached values have all metric-side artifacts they need."""
        return True

    @abstractmethod
    def evaluate(
        self,
        sample: Sample,
        layout: AbstractEvaluationLayout,
    ) -> MetricResult:
        """Compute values for one sample."""
        raise NotImplementedError
