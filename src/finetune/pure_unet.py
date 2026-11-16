"""Pure convolutional U-Net architecture used as an external baseline.

The model maps a 10-band, 256 x 256 image to a single per-pixel prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Blocks ====================
class ConvGnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, groups=16):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=16):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(g, out_ch)
        )

        if in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(g, out_ch)
            )
        else:
            self.short = nn.Identity()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = self.short(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


# ==================== Pure UNet ====================
class PureUNet(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()

        self.enc1 = nn.Sequential(
            ResidualBlock(in_ch, 64),
            ResidualBlock(64, 64),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 768),
            ResidualBlock(768, 768),
        )

        self.dec4 = nn.Sequential(
            ResidualBlock(768 + 512, 512),
            ResidualBlock(512, 512),
        )
        self.dec3 = nn.Sequential(
            ResidualBlock(512 + 256, 256),
            ResidualBlock(256, 256),
        )
        self.dec2 = nn.Sequential(
            ResidualBlock(256 + 128, 128),
            ResidualBlock(128, 128),
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(128 + 64, 64),
            ResidualBlock(64, 32),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32

        b = self.bottleneck(self.pool(e4))  # 16

        d4 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d1 = self.dropout(d1)
        return self.final(d1)
