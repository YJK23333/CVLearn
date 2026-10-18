import torch
import torch.nn as nn

import config


def _make_activation():
    return nn.ReLU(inplace=config.ACTIVATION_INPLACE)


class DoubleConv(nn.Module):
    """Two convolution layers used throughout the U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = []
        current_channels = in_channels

        for _ in range(config.CONV_LAYERS_PER_BLOCK):
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=config.CONV_KERNEL_SIZE,
                    stride=config.CONV_STRIDE,
                    padding=config.CONV_PADDING,
                    bias=config.USE_CONV_BIAS,
                )
            )
            if config.USE_BATCH_NORM:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(_make_activation())
            current_channels = out_channels

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Configurable U-Net for binary or multi-class segmentation."""

    def __init__(
        self,
        in_channels=config.IMAGE_CHANNELS,
        num_classes=config.NUM_CLASSES,
        base_channels=config.BASE_CHANNELS,
    ):
        super().__init__()

        multipliers = config.CHANNEL_MULTIPLIERS
        if len(multipliers) < 2:
            raise ValueError("CHANNEL_MULTIPLIERS must define encoder and bottleneck widths.")

        channels = [base_channels * multiplier for multiplier in multipliers]
        self.num_classes = num_classes

        # Encoder
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for out_channels in channels[:-1]:
            self.encoders.append(DoubleConv(current_channels, out_channels))
            current_channels = out_channels

        self.pool = nn.MaxPool2d(
            kernel_size=config.POOL_KERNEL_SIZE,
            stride=config.POOL_STRIDE,
        )

        # Bottleneck
        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        # Decoder: each stage upsamples, concatenates skip features, then fuses them.
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.upsamplers.append(
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=skip_channels,
                    kernel_size=config.UPSAMPLE_KERNEL_SIZE,
                    stride=config.UPSAMPLE_STRIDE,
                )
            )
            self.decoders.append(
                DoubleConv(skip_channels * 2, skip_channels)
            )
            current_channels = skip_channels

        self.out_conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=num_classes,
            kernel_size=config.OUTPUT_KERNEL_SIZE,
            stride=config.OUTPUT_STRIDE,
            padding=config.OUTPUT_PADDING,
        )

    def forward(self, x):
        skip_features = []

        # Encoder: keep feature maps for skip connections.
        for encoder in self.encoders:
            x = encoder(x)
            skip_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder: traverse skips from deepest to shallowest.
        for upsampler, decoder, skip in zip(
            self.upsamplers,
            self.decoders,
            reversed(skip_features),
        ):
            x = upsampler(x)

            # For this implementation, input height and width should be divisible
            # by 2**number_of_downsampling_stages, which the default 256x256 satisfies.
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    "Input spatial dimensions are incompatible with U-Net pooling. "
                    f"Upsampled size={tuple(x.shape[-2:])}, "
                    f"skip size={tuple(skip.shape[-2:])}. "
                    "Choose dimensions divisible by the total downsampling factor."
                )

            x = torch.cat((skip, x), dim=1)
            x = decoder(x)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet()
    sample = torch.randn(
        config.BATCH_SIZE,
        config.IMAGE_CHANNELS,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
    )
    with torch.no_grad():
        prediction = model(sample)
    print(f"Input shape:  {tuple(sample.shape)}")
    print(f"Output shape: {tuple(prediction.shape)}")
