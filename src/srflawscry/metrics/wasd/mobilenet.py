import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import mobilenet_v2


class MobileNet(nn.Module):
    """
    EfficientNet model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """

    def __init__(
        self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True
    ):
        super().__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        else:
            weights = None

        original_model = mobilenet_v2(weights=weights)

        first_block = original_model.features[0]
        first_conv = first_block[0]

        # Create a new first convolutional layer with stride=1 and padding=1
        new_first_conv = nn.Conv2d(
            in_channels=first_conv.in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,
            padding=1,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
        )

        # Copy weights from the original layer
        if self.pretrained:
            new_first_conv.weight.data = first_conv.weight.data
            if first_conv.bias is not None:
                new_first_conv.bias.data = first_conv.bias.data

        new_first_block = nn.Sequential(
            new_first_conv,
            *list(first_block)[1:],
        )

        features = list(original_model.features)
        features[0] = new_first_block

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = nn.Sequential(self.features, self.pool)

        self.feat_dim = original_model.last_channel

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim),
        )

    def forward(self, x):
        feat = self.features(x)
        f = self.pool(feat).view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        return f, g
