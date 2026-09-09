from __future__ import annotations

import torch
import torch.nn.functional as F


def _starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def _flush(
    metric,
    image_patches: list[torch.Tensor],
    reference_patches: list[torch.Tensor],
    locations: list[tuple[int, int, int, int]],
    scores: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if not image_patches:
        return
    values = metric(
        torch.stack(image_patches),
        torch.stack(reference_patches),
        require_grad=False,
        batch_average=False,
    ).reshape(-1)
    for value, (top, left, bottom, right) in zip(values, locations, strict=True):
        scores[..., top:bottom, left:right] += value
        weights[..., top:bottom, left:right] += 1.0
    image_patches.clear()
    reference_patches.clear()
    locations.clear()


@torch.inference_mode()
def compute_dists_map(
    metric,
    image: torch.Tensor,
    reference: torch.Tensor,
    patch_size: int = 64,
    stride: int = 32,
    batch_size: int = 32,
) -> torch.Tensor:
    """Reproduce the normalized block-wise DISTS input used for training."""
    if image.shape != reference.shape or image.ndim != 4:
        raise ValueError("DISTS inputs must have equal BCHW shapes")

    maps = []
    for image_item, reference_item in zip(image, reference, strict=True):
        _, height, width = image_item.shape
        scores = image_item.new_zeros((1, height, width))
        weights = image_item.new_zeros((1, height, width))
        image_patches = []
        reference_patches = []
        locations = []

        for top in _starts(height, patch_size, stride):
            for left in _starts(width, patch_size, stride):
                bottom = min(top + patch_size, height)
                right = min(left + patch_size, width)
                image_patch = image_item[:, top:bottom, left:right]
                reference_patch = reference_item[:, top:bottom, left:right]
                if image_patch.shape[-2:] != (patch_size, patch_size):
                    image_patch = F.interpolate(
                        image_patch.unsqueeze(0),
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    reference_patch = F.interpolate(
                        reference_patch.unsqueeze(0),
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                image_patches.append(image_patch)
                reference_patches.append(reference_patch)
                locations.append((top, left, bottom, right))
                if len(image_patches) >= batch_size:
                    _flush(
                        metric,
                        image_patches,
                        reference_patches,
                        locations,
                        scores,
                        weights,
                    )
        _flush(
            metric,
            image_patches,
            reference_patches,
            locations,
            scores,
            weights,
        )

        result = scores / weights.clamp_min(1.0)
        # Match the training-time precomputation, which normalizes on CPU.
        flat = result.detach().float().cpu().flatten()
        if flat.numel() > 1_000_000:
            indices = torch.linspace(
                0,
                flat.numel() - 1,
                1_000_000,
                dtype=torch.long,
            )
            flat = flat[indices]
        scale = torch.quantile(flat, 0.95).clamp_min(1e-6).item()
        maps.append((result / scale).clamp(0.0, 1.0))

    return torch.stack(maps)
