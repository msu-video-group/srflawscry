import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0


class EfficientNet(nn.Module):
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
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        original_model = efficientnet_b0(weights=weights)

        # Get the first Conv2dNormActivation block
        first_block = original_model.features[0]

        # Extract components from the first block
        first_conv = first_block[0]
        norm_layer = first_block[1] if len(first_block) > 1 else None
        activation_layer = first_block[2] if len(first_block) > 2 else None

        # Create a new first convolutional layer with stride=1
        new_first_conv = nn.Conv2d(
            in_channels=first_conv.in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
        )

        # Copy weights from the original layer
        if self.pretrained:
            new_first_conv.weight.data = first_conv.weight.data
            if first_conv.bias is not None:
                new_first_conv.bias.data = first_conv.bias.data

        # Reconstruct the first block
        new_first_block_components = [new_first_conv]

        # Add norm layer if it exists
        if norm_layer is not None:
            new_first_block_components.append(norm_layer)

        # Add activation layer if it exists
        if activation_layer is not None:
            new_first_block_components.append(activation_layer)

        new_first_block = nn.Sequential(*new_first_block_components)

        # Replace the first block in features
        features = list(original_model.features)
        features[0] = new_first_block

        self.model = nn.Sequential(
            *features,
            original_model.avgpool,
            nn.Flatten(),
        )

        self.feat_dim = original_model.classifier[1].in_features

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim),
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        return f, g
