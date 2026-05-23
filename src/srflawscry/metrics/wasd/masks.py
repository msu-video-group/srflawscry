import cv2
import numpy as np
import torch


def largest_connected_component(
    mask: np.ndarray,
    min_area: int = 1,
    min_area_fraction: float = 0.0,
) -> np.ndarray:
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(binary)

    min_area = max(min_area, int(binary.size * min_area_fraction))
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(component_areas.argmax()) + 1
    if component_areas[largest_label - 1] < min_area:
        return np.zeros_like(binary)

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def filter_connected_components(
    mask: np.ndarray,
    min_area: int = 1,
    min_area_fraction: float = 0.0,
) -> np.ndarray:
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(binary)

    min_area = max(min_area, int(binary.size * min_area_fraction))
    filtered = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered.astype(np.uint8)


def largest_connected_component_tensor(
    mask: torch.Tensor,
    min_area: int = 1,
    min_area_fraction: float = 0.0,
) -> torch.Tensor:
    mask_np = mask.detach().squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    primary_mask = largest_connected_component(
        mask_np,
        min_area=min_area,
        min_area_fraction=min_area_fraction,
    )

    primary_tensor = torch.from_numpy(primary_mask).to(
        device=mask.device, dtype=mask.dtype
    )
    primary_tensor = primary_tensor / 255
    while primary_tensor.ndim < mask.ndim:
        primary_tensor = primary_tensor.unsqueeze(0)

    return primary_tensor


def filter_connected_components_tensor(
    mask: torch.Tensor,
    min_area: int = 1,
    min_area_fraction: float = 0.0,
) -> torch.Tensor:
    mask_np = mask.detach().squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    filtered_mask = filter_connected_components(
        mask_np,
        min_area=min_area,
        min_area_fraction=min_area_fraction,
    )

    filtered_tensor = torch.from_numpy(filtered_mask).to(
        device=mask.device, dtype=mask.dtype
    )
    filtered_tensor = filtered_tensor / 255
    while filtered_tensor.ndim < mask.ndim:
        filtered_tensor = filtered_tensor.unsqueeze(0)

    return filtered_tensor
