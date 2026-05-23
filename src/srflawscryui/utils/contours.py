from collections.abc import Iterable
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from srflawscry.utils.images import open_rgb_image


def draw_contours(
    img_path: Path,
    mask_paths: Iterable[Path],
) -> np.ndarray:
    try:
        img = np.asarray(open_rgb_image(img_path)).copy()
    except Exception as error:
        raise gr.Error(f"Failed to load image: {img_path}") from error
    if img is None:
        raise gr.Error(f"Failed to load image: {img_path}")

    for mask_path in mask_paths:
        mask = _read_binary_mask(mask_path)
        if np.count_nonzero(mask) == 0:
            continue
        mask = _match_image_size(mask, img)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 7)

    return img


def _read_binary_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros((1, 1), dtype=np.uint8)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _match_image_size(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    image_h, image_w = img.shape[:2]
    if mask.shape == (image_h, image_w):
        return mask
    return cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
