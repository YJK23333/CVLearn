
import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights

import config


class VGG16Backbone(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        weights = (
            VGG16_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = vgg16(weights=weights)

        # Keep convolutional layers through Conv5_3.
        self.features = nn.Sequential(
            *list(model.features.children())[:30]
        )

        # Freeze early convolutional layers.
        for parameter in list(
            self.features.parameters()
        )[:]:
            parameter.requires_grad = True

        self.out_channels = config.FEATURE_CHANNELS

    def forward(self, x):

        return self.features(x)
