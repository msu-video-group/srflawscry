from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    item_id: str
    result_path: Path


@dataclass(frozen=True)
class SampleWithSource(Sample):
    source_path: Path


@dataclass(frozen=True)
class SampleWithGroundTruth(Sample):
    ground_truth_path: Path
