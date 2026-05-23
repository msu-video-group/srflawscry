import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from dotmap import DotMap

from .efficientnet import EfficientNet
from .mobilenet import MobileNet
from .nafnet import NAFNetRefiner
from .resnet import ResNet

logger = logging.getLogger(__name__)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionRefiner(torch.nn.Module):
    """
    UNet-like refiner that takes image and first stage mask and outputs refined mask.
    Input:
        - image: [B, C, H, W]
        - coarse_mask: [B, 1, H, W]
    Output:
        - refined_mask: [B, 1, H, W]
    """

    def __init__(self, in_image_channels=3, base_channels=32):
        super().__init__()
        in_ch = in_image_channels  # + 1

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = torch.nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = torch.nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.final_conv = torch.nn.Conv2d(base_channels, 1, kernel_size=1)

        self.se_fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(base_channels * 8, base_channels * 8 // 8, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 8 // 8, base_channels * 8, kernel_size=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, image, reference, coarse_mask):
        if isinstance(coarse_mask, (list, tuple)):
            coarse_mask = torch.cat(coarse_mask, dim=1)

        # x = torch.cat([image, reference], dim=1)
        x = torch.cat([image, reference, coarse_mask], dim=1)

        e1 = self.enc1(x)  # H/1
        e2 = self.enc2(F.avg_pool2d(e1, 2))  # H/2
        e3 = self.enc3(F.avg_pool2d(e2, 2))  # H/4

        b = self.bottleneck(F.avg_pool2d(e3, 2))  # H/8

        se = self.se_fc(b)
        b = b * se

        d3 = self.up3(b)  # H/4

        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # H/2
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # H/1
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = torch.sigmoid(out)
        return out


class TripletModel(torch.nn.Module):
    """
    Triplet model class for learning with anchor, positive, and negative examples,
    with additional MLP head for prominence prediction.

    Args:
        encoder_params (dict / DotMap): encoder parameters with keys
            - embedding_dim (int): embedding dimension of the encoder projection head
            - pretrained (bool): whether to use pretrained weights for the encoder
            - use_norm (bool): whether normalize the embeddings
        margin (float): margin for the triplet loss. Default: 1.0
        prominence_head_hidden (int): hidden dimension for prominence MLP head. Default: 256
    """

    def __init__(
        self,
        encoder_params: DotMap,
        margin: float = 1.0,
        prominence_head_hidden: int = 256,
        loss_weights=None,
    ):
        super().__init__()

        supported_encoders = ["resnet", "effnet", "mobnet"]
        if encoder_params.encoder == "resnet":
            self.encoder = ResNet(
                embedding_dim=encoder_params.embedding_dim,
                pretrained=encoder_params.pretrained,
                use_norm=encoder_params.use_norm,
            )
        elif encoder_params.encoder == "effnet":
            self.encoder = EfficientNet(
                embedding_dim=encoder_params.embedding_dim,
                pretrained=encoder_params.pretrained,
                use_norm=encoder_params.use_norm,
            )
        elif encoder_params.encoder == "mobnet":
            self.encoder = MobileNet(
                embedding_dim=encoder_params.embedding_dim,
                pretrained=encoder_params.pretrained,
                use_norm=encoder_params.use_norm,
            )
        else:
            raise ValueError(
                f"Unknown encoder name: {encoder_params.encoder}. Supported: {supported_encoders}"
            )

        self.margin = margin
        self.criterion = triplet_loss
        self.projection_embedding_size = self.encoder.feat_dim

        # Add MLP head for prominence prediction
        self.prominence_head = torch.nn.Sequential(
            torch.nn.Linear(self.projection_embedding_size, prominence_head_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(prominence_head_hidden, prominence_head_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(prominence_head_hidden, prominence_head_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(prominence_head_hidden, 1),
        )

        self.mse_loss = torch.nn.MSELoss()

        self.extra_mask_dirs = encoder_params.extra_mask_dirs
        if isinstance(self.extra_mask_dirs, DotMap):
            self.extra_mask_dirs = []
        if isinstance(self.extra_mask_dirs, str):
            self.extra_mask_dirs = [self.extra_mask_dirs]

        refiner_channels = 7 + len(self.extra_mask_dirs)

        if encoder_params.refiner == "unet":
            self.refiner = AttentionRefiner(
                in_image_channels=refiner_channels, base_channels=32
            )
        elif encoder_params.refiner == "nafnet":
            self.refiner = NAFNetRefiner(
                in_channels=refiner_channels,
                width=32,
                pretrained=encoder_params.nafnet_pretrained,
            )

        if loss_weights is None:
            self.loss_weights = {
                "mse": 0.0,
                "triplet_gt": 1.0,
                "triplet_no_art": 1.0,
                "triplet_outside": 1.0,
            }
        else:
            self.loss_weights = loss_weights

        logger.info(f"Loss weights: {self.loss_weights}")

    def freeze_encoder(self, freeze: bool = True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

        if freeze:
            logger.info("Freeze encoder weights")
        else:
            logger.info("Unfreeze encoder weights")

    def freeze_prominence_head(self, freeze: bool = True):
        for param in self.prominence_head.parameters():
            param.requires_grad = not freeze

        if freeze:
            logger.info("Freeze prominence head")
        else:
            logger.info("Unfreeze prominence head")

    def freeze_refiner(self, freeze: bool = True):
        for param in self.refiner.parameters():
            param.requires_grad = not freeze
        logger.info("Freeze refiner" if freeze else "Unfreeze refiner")

    def forward(
        self,
        anchor,
        positive=None,
        negative_gt=None,
        negative_no_artifacts=None,
        negative_outside_mask=None,
        prominence=-1,
        calculate_loss_in_val=False,
        return_prominence=False,
        reference=None,
        artifact_mask_gt=None,
        artifact_crop_size=256,
        name=None,
    ):

        if reference is None:
            return self.forward_contrastive(
                anchor,
                positive,
                negative_gt,
                negative_no_artifacts,
                negative_outside_mask,
                prominence,
                calculate_loss_in_val,
                return_prominence,
            )
        return self.forward_refiner(
            anchor, reference, artifact_mask_gt, artifact_crop_size, name
        )

    def forward_refiner(
        self, image, reference, artifact_mask_gt=None, artifact_crop_size=256, name=None
    ):

        device = image.device

        with torch.no_grad():
            coarse_mask = self.generate_artifact_mask(
                image, reference, artifact_crop_size, device
            )

        extra_masks = torch.empty(0).to(device)
        if self.extra_mask_dirs and name:
            """extra_masks = []
            for tensor_name in name:
                tensor_masks = []
                for dirname in self.extra_mask_dirs:
                    try:
                        tensor_masks.append(torch.from_numpy(np.load(dirname + '/' + tensor_name + '.npy')))
                    except:
                        prefixed_files = [filename for filename in os.listdir(dirname)
                            if filename.startswith(tensor_name)]
                        try:
                            tensor_masks.append(torch.from_numpy(np.load(dirname + '/' + prefixed_files[0])))
                        except:
                            tensor_masks.append((0,))
                extra_masks.append(torch.cat(tensor_masks, dim=1))
            extra_masks = torch.stack(extra_masks).to(coarse_mask.device)"""

            dir_files = {}
            for dirname in self.extra_mask_dirs:
                p = Path(dirname)
                files = list(p.iterdir())
                dir_files[dirname] = {
                    "exact": {f.stem: f for f in files if f.suffix == ".npy"},
                    "all": files,
                }

            extra_masks = []

            for tensor_name in name:
                tensor_masks = []

                for dirname in self.extra_mask_dirs:
                    entry = dir_files[dirname]
                    file_path = entry["exact"].get(tensor_name)

                    if file_path is None:
                        file_path = next(
                            (
                                f
                                for f in entry["all"]
                                if f.name.startswith(tensor_name) and f.suffix == ".npy"
                            ),
                            None,
                        )

                    if file_path is not None:
                        arr = np.load(file_path, allow_pickle=False)
                        tensor_masks.append(torch.from_numpy(arr))
                    else:
                        tensor_masks.append(torch.zeros(1))

                extra_masks.append(torch.cat(tensor_masks, dim=1))

            extra_masks = torch.stack(extra_masks, dim=0).to(coarse_mask.device)

            while extra_masks.ndim < coarse_mask.ndim:
                extra_masks = extra_masks.unsqueeze(0)

        refined_mask = self.refiner(
            image, reference, [coarse_mask.detach(), extra_masks]
        )

        return refined_mask, coarse_mask

    def forward_contrastive(
        self,
        anchor,
        positive=None,
        negative_gt=None,
        negative_no_artifacts=None,
        negative_outside_mask=None,
        prominence=-1,
        calculate_loss_in_val=False,
        return_prominence=False,
    ):

        anchor_emb, anchor_proj = self.encoder(anchor)

        if return_prominence:
            prominence_pred = self.prominence_head(anchor_emb).squeeze(-1)
            return anchor_emb, anchor_proj, prominence_pred

        if not self.training and not calculate_loss_in_val:
            # During inference, return embeddings and prominence prediction
            prominence_pred = self.prominence_head(anchor_emb).squeeze(-1)
            return anchor_emb, anchor_proj, prominence_pred
        if positive is None:
            prominence_pred = self.prominence_head(anchor_emb).squeeze(-1)
            mse_loss = self.mse_loss(prominence_pred, prominence.float())
            return mse_loss

        assert positive is not None and negative_gt is not None, (
            "You should pass positive and negative examples during training"
        )
        assert torch.all(prominence >= 0), (
            "You should pass prominence scores during training"
        )

        positive_emb, positive_proj = self.encoder(positive)
        negative_emb, negative_proj = self.encoder(negative_gt)

        # Calculate triplet loss
        if negative_no_artifacts is not None and negative_outside_mask is not None:
            negative_no_artifacts_emb, negative_no_artifacts_proj = self.encoder(
                negative_no_artifacts
            )
            negative_outside_mask_emb, negative_outside_mask_proj = self.encoder(
                negative_outside_mask
            )
            triplet_loss, triplet_loss_parts = self.criterion(
                anchor_proj,
                positive_proj,
                negative_proj,
                negative_no_artifacts_proj,
                negative_outside_mask_proj,
                self.margin,
            )
        else:
            triplet_loss, triplet_loss_parts = self.criterion(
                anchor_proj, positive_proj, negative_proj, None, self.margin
            )

        # Calculate prominence prediction loss
        prominence_pred = self.prominence_head(anchor_emb).squeeze(-1)
        mse_loss = self.mse_loss(prominence_pred, prominence.float())

        # Combine losses
        total_loss = (
            self.loss_weights["mse"] * mse_loss
            + self.loss_weights["triplet_gt"] * triplet_loss_parts[0]
            + self.loss_weights["triplet_no_art"] * triplet_loss_parts[1]
            + self.loss_weights["triplet_outside"] * triplet_loss_parts[2]
        )

        return total_loss, triplet_loss_parts, mse_loss

    def generate_artifact_mask(self, images, reference, crop_size, device):
        """
        Generate artifact mask by comparing image blocks with corresponding  reference blocks
        using cosine similarity between their embeddings.

        Args:
            images (torch.Tensor): Input images tensor of shape [B, C, H, W]
            reference (torch.Tensor): Reference GT images tensor of shape [B, C, H, W]
            crop_size (int): Size of the blocks to process
            device: Current device

        Returns:
            torch.Tensor: Artifact mask of shape [B, 1, H, W] with values in [0, 1]
        """
        batch_size, channels, height, width = images.shape

        num_blocks_h = height // crop_size
        num_blocks_w = width // crop_size

        artifact_mask = torch.zeros(batch_size, 1, height, width, device=device)

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                h_start = i * crop_size
                h_end = h_start + crop_size
                w_start = j * crop_size
                w_end = w_start + crop_size

                image_blocks = images[:, :, h_start:h_end, w_start:w_end]
                reference_blocks = reference[:, :, h_start:h_end, w_start:w_end]

                with torch.no_grad():
                    image_embs, _ = self.encoder(image_blocks)
                    reference_embs, _ = self.encoder(reference_blocks)

                image_embs_norm = F.normalize(image_embs, p=2, dim=1)
                reference_embs_norm = F.normalize(reference_embs, p=2, dim=1)
                cosine_similarity = torch.sum(
                    image_embs_norm * reference_embs_norm, dim=1
                )

                # cos = 1 -> 0 (no artifact)
                # cos = -1 -> 1 (artifact)
                patch_score = (1 - cosine_similarity) / 2

                patch_score = patch_score.view(batch_size, 1, 1, 1)
                artifact_mask[:, :, h_start:h_end, w_start:w_end] = patch_score

        artifact_mask = torch.clamp(artifact_mask, 0.0, 1.0)
        return artifact_mask


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative_gt: torch.Tensor,
    negative_no_artifact: torch.Tensor = None,
    negative_outside_mask: torch.Tensor = None,
    margin: float = 1.0,
):
    """
    Compute the triplet loss with one positive and two negative examples.

    Args:
        anchor (torch.Tensor): anchor embeddings
        positive (torch.Tensor): positive embeddings
        negative_gt (torch.Tensor): negative embeddings
        negative_no_artifact (torch.Tensor): second negative embeddings (optional)
        negative_outside_mask (torch.Tensor): third negative embeddings (optional)
        margin (float): margin for the loss
    """
    anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)
    positive_norm = torch.nn.functional.normalize(positive, p=2, dim=1)
    negative_gt_norm = torch.nn.functional.normalize(negative_gt, p=2, dim=1)

    pos_dist = torch.sum((anchor_norm - positive_norm).pow(2), dim=1)
    neg_dist = torch.sum((anchor_norm - negative_gt_norm).pow(2), dim=1)

    basic_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)

    if negative_no_artifact is not None and negative_outside_mask is not None:
        negative_no_artifact_norm = torch.nn.functional.normalize(
            negative_no_artifact, p=2, dim=1
        )
        neg_no_artifact_dist = torch.sum(
            (anchor_norm - negative_no_artifact_norm).pow(2), dim=1
        )

        negative_outside_mask_norm = torch.nn.functional.normalize(
            negative_outside_mask, p=2, dim=1
        )
        negative_outside_mask_dist = torch.sum(
            (anchor_norm - negative_outside_mask_norm).pow(2), dim=1
        )

        # Second triplet loss component with second negative
        additional_loss = torch.clamp(pos_dist - neg_no_artifact_dist + margin, min=0.0)
        additional_loss_negative = torch.clamp(
            pos_dist - negative_outside_mask_dist + margin, min=0.0
        )

        loss = (basic_loss + additional_loss + additional_loss_negative) / 3
    else:
        loss = basic_loss
        additional_loss = 0.0
        additional_loss_negative = 0.0

    return torch.mean(loss), [
        torch.mean(basic_loss),
        torch.mean(additional_loss),
        torch.mean(additional_loss_negative),
    ]
