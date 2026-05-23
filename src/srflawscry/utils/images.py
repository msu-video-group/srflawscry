from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

register_heif_opener()


def open_rgb_image(path: Path | str) -> Image.Image:
    path = Path(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def read_image_tensor(path: Path | str) -> torch.Tensor:
    image = open_rgb_image(path)
    data = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(data.copy()).permute(2, 0, 1).contiguous()
