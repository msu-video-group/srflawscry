from pathlib import Path

import cv2

from srflawscry.metrics.wasd import WASD
from srflawscry.utils.device import preferred_device


def test_wasd(tmp_path):
    image_path = Path("assets/data/RealESRGAN/results/00000462.png")
    source_path = Path("assets/data/RealESRGAN/sources/00000462.png")

    output_path = tmp_path / "wasd.png"
    detector = WASD.from_pretrained("egorchistov/sr-artifact-detection-wasd")
    detector = detector.eval().to(preferred_device())

    detector.run(image_path, source_path, output_path)

    mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.sum() > 0
