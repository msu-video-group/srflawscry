import cv2
import torch

from srflawscry.metrics.wasd import WASD
from srflawscry.metrics.wasd.dists_map import compute_dists_map
from srflawscry.utils.device import preferred_device
from srflawscryui.demo_data import demo_data_root


class MeanAbsoluteDifference(torch.nn.Module):
    def forward(self, x, y, require_grad=False, batch_average=False):
        _ = require_grad
        values = (x - y).abs().mean(dim=(1, 2, 3))
        return values.mean() if batch_average else values


def test_dists_map_shape_and_range():
    image = torch.zeros(2, 3, 80, 96)
    reference = image.clone()
    image[0, :, 20:60, 20:60] = 1
    image[1, :, 40:, 48:] = 0.5

    result = compute_dists_map(
        MeanAbsoluteDifference(),
        image,
        reference,
        patch_size=64,
        stride=32,
        batch_size=3,
    )

    assert result.shape == (2, 1, 80, 96)
    assert torch.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1
    assert result.sum() > 0


def test_wasd(tmp_path):
    data_root = demo_data_root()
    image_path = data_root / "RealESRGAN/results/00000462.png"
    source_path = data_root / "RealESRGAN/sources/00000462.png"

    output_path = tmp_path / "wasd.png"
    detector = WASD.from_pretrained("egorchistov/sr-artifact-detection-wasd")
    detector = detector.eval().to(preferred_device())

    detector.run(image_path, source_path, output_path)

    mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.sum() > 0
