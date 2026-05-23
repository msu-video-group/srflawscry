from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from srflawscry.samples import Sample


class AbstractEvaluationLayout(ABC):
    """Dataset layout; implement sample discovery and metric output paths."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable evaluated set name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metrics_path(self) -> Path:
        """CSV cache path used by MetricRunner."""
        raise NotImplementedError

    @abstractmethod
    def iter_samples(self) -> list[Sample]:
        """Return samples available in this layout."""
        raise NotImplementedError

    @abstractmethod
    def metric_output_path(
        self,
        metric_id: str,
        output_name: str,
        item_id: str,
    ) -> Path:
        """Return where a metric should store one generated artifact."""
        raise NotImplementedError

    def sample(self, item_id: str) -> Sample:
        samples = {sample.item_id: sample for sample in self.iter_samples()}
        if item_id not in samples:
            raise KeyError(f"Item not found in {self.name}: {item_id}")
        return samples[item_id]
