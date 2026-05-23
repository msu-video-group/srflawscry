from __future__ import annotations

import torch


def preferred_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()
