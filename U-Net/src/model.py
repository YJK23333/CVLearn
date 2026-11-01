
import torch
import torch.nn as nn
import torch.nn.functional as F

import config



class DoubleConv(nn.Module):
    """
    Original U-Net style:
    Conv2d -> ReLU -> Conv2d -> ReLU
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=64
    ):

        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            c * 16, c * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(
            c * 8, c * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(
            c * 4, c * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(
            c * 2, c, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(c * 2, c)

        # Output
        self.out_conv = nn.Conv2d(
            c, out_channels, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + Skip Connections
        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # Return logits, not sigmoid probabilities
        output = self.out_conv(d1)

        return output


def build_model():

    model = UNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    )

    return model
