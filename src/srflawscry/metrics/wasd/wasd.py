import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from dotmap import DotMap
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from srflawscry.utils.device import empty_device_cache
from srflawscry.utils.images import read_image_tensor

from .masks import filter_connected_components_tensor
from .triplet_loss_model import TripletModel


class WASD(nn.Module, PyTorchModelHubMixin):
    """WASD: Weakly-supervised Artifact Scoring with Discriminative triplet loss."""

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
        use_norm: bool = True,
        margin: float = 1.0,
        prominence_head_hidden: int = 256,
        encoder: str = "mobnet",
        refiner: str = "unet",
        crop_size: int = 100,
    ) -> None:
        super().__init__()

        model_config = DotMap()
        model_config.embedding_dim = embedding_dim
        model_config.pretrained = pretrained
        model_config.use_norm = use_norm
        model_config.encoder = encoder
        model_config.refiner = refiner

        model = TripletModel(
            encoder_params=model_config,
            margin=margin,
            prominence_head_hidden=prominence_head_hidden,
        )

        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.crop_size = crop_size
        self.artifact_threshold = 0.92
        self.min_mask_area_fraction = 0.0005
        self.max_inference_size = 1024

    def close(self) -> None:
        self.cpu()
        gc.collect()
        empty_device_cache()

    def forward(
        self,
        uint8_rgb_xN_image: torch.LongTensor,
        uint8_rgb_x1_source: torch.LongTensor,
        threshold: float | None = None,
    ) -> torch.FloatTensor:
        threshold = self.artifact_threshold if threshold is None else threshold
        image = self.transform(uint8_rgb_xN_image).unsqueeze(0).to(self.device)
        source = self.transform(uint8_rgb_x1_source).unsqueeze(0).to(self.device)

        if image.shape[-2] % source.shape[-2] or image.shape[-1] % source.shape[-1]:
            raise ValueError(
                f"upscaled image must be N times larger than the source image, "
                f"but got {image.shape[-2]}x{image.shape[-1]} upscaled image and "
                f"{source.shape[-2]}x{source.shape[-1]} source image",
            )
        if image.shape[-2] // source.shape[-2] != image.shape[-1] // source.shape[-1]:
            raise ValueError(
                "upscaled image and source image have different aspect ratios"
            )
        if image.shape[-2] // source.shape[-2] == 1:
            raise ValueError("upscaled image and source image are the same size")
        scale_factor = image.shape[-2] // source.shape[-2]

        reference = F.interpolate(
            source,
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        confidence, _ = self.model(
            image,
            reference=reference,
            artifact_crop_size=self.crop_size,
        )

        mask = (confidence > threshold).float()

        return mask.squeeze(0)

    @torch.inference_mode()
    def run(
        self,
        image_path: Path,
        source_path: Path,
        output_path: Path,
        postprocess: bool = True,
        threshold: float | None = None,
        min_mask_area_fraction: float | None = None,
        max_inference_size: int | None = None,
    ) -> None:
        image = read_image_tensor(image_path)
        source = read_image_tensor(source_path)

        original_size = image.shape[-2:]
        image, source = self._resize_for_inference(
            image,
            source,
            max_inference_size or self.max_inference_size,
        )

        try:
            binary_mask = self(image, source, threshold=threshold)
        except ValueError as error:
            raise ValueError(
                "Failed to run WASD for "
                f"result image {image_path} and source image {source_path}: {error}"
            ) from error
        if binary_mask.shape[-2:] != original_size:
            binary_mask = TF.resize(
                binary_mask,
                list(original_size),
                interpolation=InterpolationMode.NEAREST,
            )
        if postprocess:
            binary_mask = filter_connected_components_tensor(
                binary_mask,
                min_area_fraction=(
                    self.min_mask_area_fraction
                    if min_mask_area_fraction is None
                    else min_mask_area_fraction
                ),
            )

        save_image(binary_mask, output_path)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _resize_for_inference(
        image: torch.Tensor,
        source: torch.Tensor,
        max_inference_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not max_inference_size or max(image.shape[-2:]) <= max_inference_size:
            return image, source

        image_h, image_w = image.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if image_h % source_h or image_w % source_w:
            return image, source

        scale_h = image_h // source_h
        scale_w = image_w // source_w
        if scale_h != scale_w:
            return image, source

        scale = scale_h
        resize_factor = max_inference_size / max(image_h, image_w)
        resized_h = max(scale, int(image_h * resize_factor) // scale * scale)
        resized_w = max(scale, int(image_w * resize_factor) // scale * scale)
        if resized_h == image_h and resized_w == image_w:
            return image, source

        resized_image = TF.resize(
            image,
            [resized_h, resized_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized_source = TF.resize(
            source,
            [resized_h // scale, resized_w // scale],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return resized_image, resized_source
