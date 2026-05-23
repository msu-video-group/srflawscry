from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from srflawscry.layouts.abstract import AbstractEvaluationLayout
from srflawscry.samples import (
    SampleWithSource,
)
from srflawscry.utils.images import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class FolderWithSourcesLayoutConfig:
    result_dir: str = "results"
    source_dir: str = "sources"
    output_dir: str = "outputs"
    metrics_filename: str = "metrics.csv"


DEFAULT_FOLDER_WITH_SOURCES_LAYOUT_CONFIG = FolderWithSourcesLayoutConfig()


class FolderWithSourcesLayout(AbstractEvaluationLayout):
    def __init__(
        self,
        root: Path,
        config: FolderWithSourcesLayoutConfig = DEFAULT_FOLDER_WITH_SOURCES_LAYOUT_CONFIG,
        name: str = "",
    ) -> None:
        self.root = root
        self.config = config
        self._name = name if name else root.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics_path(self) -> Path:
        return self.root / self.config.metrics_filename

    def iter_samples(self) -> list[SampleWithSource]:
        result_root = self.root / self.config.result_dir
        if not result_root.is_dir():
            return []

        samples = []
        for result_path in sorted(result_root.iterdir()):
            if result_path.is_file() and result_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(self._sample(result_path))
        return samples

    def metric_output_path(
        self,
        metric_id: str,
        output_name: str,
        item_id: str,
    ) -> Path:
        output_id = f"{Path(item_id).stem}.png"
        return self.root / self.config.output_dir / metric_id / output_name / output_id

    def _sample(self, result_path: Path) -> SampleWithSource:
        item_id = result_path.name
        return SampleWithSource(
            item_id=item_id,
            result_path=result_path,
            source_path=self._source_path(result_path),
        )

    def _source_path(self, result_path: Path) -> Path:
        source_root = self.root / self.config.source_dir
        exact_path = source_root / result_path.name
        if exact_path.exists():
            return exact_path

        if not source_root.is_dir():
            return exact_path

        for source_path in sorted(source_root.iterdir()):
            if (
                source_path.is_file()
                and source_path.stem == result_path.stem
                and source_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                return source_path
        return exact_path
