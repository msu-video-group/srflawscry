from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEMO_DATASET_REPO_ID = "egorchistov/sr-artifact-detection-demo-dataset"
DEMO_DATASET_DATA_DIR = "data"
DEMO_DATASET_METHODS = ("RealESRGAN", "SwinIR")
LOCAL_ASSETS_ROOT = Path("assets")


def demo_data_root() -> Path:
    configured_root = os.getenv("SRFLAWSCRY_DEMO_DATA_DIR")
    if configured_root:
        return Path(configured_root)

    local_root = LOCAL_ASSETS_ROOT / DEMO_DATASET_DATA_DIR
    if _has_demo_data(local_root):
        return local_root

    repo_id = os.getenv("SRFLAWSCRY_DEMO_DATASET", DEMO_DATASET_REPO_ID)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{DEMO_DATASET_DATA_DIR}/**"],
        local_dir=LOCAL_ASSETS_ROOT,
    )
    return local_root


def demo_result_dirs() -> tuple[str, str]:
    root = demo_data_root()
    return tuple(str(root / method) for method in DEMO_DATASET_METHODS)


def _has_demo_data(root: Path) -> bool:
    return all((root / method).is_dir() for method in DEMO_DATASET_METHODS)
