from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from srflawscry.layouts import AbstractEvaluationLayout
from srflawscry.metrics.abstract import (
    AbstractMetric,
    MetricResult,
    MetricSpec,
)
from srflawscry.metrics.wasd.labels import (
    ARTIFACT_LABELS,
    ARTIFACT_MASK_NAME,
)
from srflawscry.samples import SampleWithSource
from srflawscry.utils.device import preferred_device


class WASDMetric(AbstractMetric):
    metric_id = "WASD"

    def __init__(
        self,
        model_id: str = "egorchistov/sr-artifact-detection-wasd",
        mask_name: str = ARTIFACT_MASK_NAME,
        tolerance: float = 0.004,
        reuse_existing_masks: bool = True,
        detector_threshold: float = 0.92,
        min_mask_area_fraction: float | None = 0.0005,
    ) -> None:
        self.model_id = model_id
        self.mask_name = mask_name
        self.reuse_existing_masks = reuse_existing_masks
        self.detector_threshold = detector_threshold
        self.min_mask_area_fraction = min_mask_area_fraction
        self._specs = tuple(
            MetricSpec(
                name=label,
                higher_is_better=False,
                tolerance=tolerance,
                mask_name=mask_name,
            )
            for label in ARTIFACT_LABELS
        )
        self._detector = None
        self._classifier = None

    @property
    def open_steps(self) -> tuple[str, ...]:
        return (
            "Loading WASD weights",
            "Loading VLM weights",
        )

    @property
    def is_open(self) -> bool:
        return self._detector is not None and self._classifier is not None

    @property
    def specs(self) -> tuple[MetricSpec, ...]:
        return self._specs

    def open(self) -> None:
        if self.is_open:
            return

        for step in self.open_steps:
            self.open_step(step)

    def open_step(self, step: str) -> None:
        if step == "Loading WASD weights":
            self._open_detector()
            return
        if step == "Loading VLM weights":
            self._open_classifier()
            return
        super().open_step(step)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        if self._classifier is not None:
            self._classifier.close()
        self._detector = None
        self._classifier = None

    def evaluate(
        self,
        sample: SampleWithSource,
        layout: AbstractEvaluationLayout,
    ) -> MetricResult:
        if self._detector is None or self._classifier is None:
            self.open()

        values = {label: 0.0 for label in ARTIFACT_LABELS}
        mask_path = layout.metric_output_path(
            self.metric_id,
            self.mask_name,
            sample.item_id,
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.reuse_existing_masks or not mask_path.exists():
            self._detector.run(
                sample.result_path,
                sample.source_path,
                mask_path,
                threshold=self.detector_threshold,
                min_mask_area_fraction=self.min_mask_area_fraction,
            )

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load artifact mask: {mask_path}")

        area = float(np.mean(mask > 127))
        if area == 0.0:
            self._write_zero_mask(mask_path, mask)
            return MetricResult(values)

        labels = self._classifier.classify(
            sample.source_path,
            sample.result_path,
            mask_path,
        )
        labels = [label for label in labels if label in values]
        if not labels:
            self._write_zero_mask(mask_path, mask)
            return MetricResult(values)

        for label in labels:
            values[label] = area

        return MetricResult(values)

    def has_cached_outputs(
        self,
        layout: AbstractEvaluationLayout,
        item_id: str,
    ) -> bool:
        mask_path = layout.metric_output_path(
            self.metric_id,
            self.mask_name,
            item_id,
        )
        return self.reuse_existing_masks and mask_path.exists()

    def _write_zero_mask(self, mask_path: Path, mask: np.ndarray) -> None:
        cv2.imwrite(str(mask_path), np.zeros_like(mask, dtype=np.uint8))

    def _open_detector(self) -> None:
        if self._detector is not None:
            return

        from srflawscry.metrics.wasd.wasd import WASD

        device = preferred_device()
        self._detector = WASD.from_pretrained(self.model_id).eval().to(device)

    def _open_classifier(self) -> None:
        if self._classifier is not None:
            return

        from srflawscry.metrics.wasd.vlm import SmolVLMArtifactClassifier

        self._classifier = SmolVLMArtifactClassifier()
        self._classifier.load()
