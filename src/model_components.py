"""Reusable layers, losses, and model wrappers for degradation mapping.

The training entry point assembles these components with the SkySense++ encoder and
LoRA adapters. Data loading and experiment orchestration live in ``train_model.py``."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== LoRA ====================
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def apply_attn_ffn_lora(backbone, rank=8, alpha=16, start_layer=0):
    """Apply LoRA to Attention and FFN across the entire backbone (start_layer=0)."""
    lora_params = []

    if hasattr(backbone, "layers"):
        blocks = backbone.layers
    elif hasattr(backbone, "blocks"):
        blocks = backbone.blocks
    else:
        raise AttributeError("Backbone has neither .layers nor .blocks")

    for i, layer in enumerate(blocks):
        if i < start_layer:
            continue

        # ---- attention qkv ----
        if (
            hasattr(layer, "attn")
            and hasattr(layer.attn, "qkv")
            and isinstance(layer.attn.qkv, nn.Linear)
        ):
            orig_qkv = layer.attn.qkv
            lora_qkv = LoRALinear(orig_qkv, rank=rank, alpha=alpha)
            layer.attn.qkv = lora_qkv
            lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])

        # ---- attention proj ----
        if (
            hasattr(layer, "attn")
            and hasattr(layer.attn, "proj")
            and isinstance(layer.attn.proj, nn.Linear)
        ):
            orig_proj = layer.attn.proj
            lora_proj = LoRALinear(orig_proj, rank=rank, alpha=alpha)
            layer.attn.proj = lora_proj
            lora_params.extend([lora_proj.lora_A, lora_proj.lora_B])

        # ---- FFN / MLP ----
        if hasattr(layer, "ffn") and hasattr(layer.ffn, "layers"):
            # standalone_vit/mmseg-style
            orig1 = layer.ffn.layers[0][0]
            lora1 = LoRALinear(orig1, rank=rank, alpha=alpha)
            layer.ffn.layers[0][0] = lora1
            lora_params.extend([lora1.lora_A, lora1.lora_B])

            orig2 = layer.ffn.layers[1]
            lora2 = LoRALinear(orig2, rank=rank, alpha=alpha)
            layer.ffn.layers[1] = lora2
            lora_params.extend([lora2.lora_A, lora2.lora_B])

        elif hasattr(layer, "mlp"):
            # timm-style fallback
            if hasattr(layer.mlp, "fc1") and isinstance(layer.mlp.fc1, nn.Linear):
                orig1 = layer.mlp.fc1
                lora1 = LoRALinear(orig1, rank=rank, alpha=alpha)
                layer.mlp.fc1 = lora1
                lora_params.extend([lora1.lora_A, lora1.lora_B])

            if hasattr(layer.mlp, "fc2") and isinstance(layer.mlp.fc2, nn.Linear):
                orig2 = layer.mlp.fc2
                lora2 = LoRALinear(orig2, rank=rank, alpha=alpha)
                layer.mlp.fc2 = lora2
                lora_params.extend([lora2.lora_A, lora2.lora_B])

    n = sum(p.numel() for p in lora_params)
    print(
        "Attn+FFN LoRA: start_layer=%d (full backbone), rank=%d, %.2fM params"
        % (start_layer, rank, n / 1e6)
    )
    return lora_params


# ==================== Decoder ====================
class ConvGnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, groups=16):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


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


class GatedFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ResidualBlock(ch, ch)

    def forward(self, x_prev, x_skip):
        g = self.gate(torch.cat([x_prev, x_skip], dim=1))
        x = g * x_prev + (1.0 - g) * x_skip
        return self.refine(x)


class UNetGatedDecoder(nn.Module):
    """
    Progressive gated-fusion decoder using 4 backbone feature levels.
    All backbone features are 64x64.
    Fusion path: c4 -> c3 -> c2 -> c1, then upsample to 128 and 256.
    """

    def __init__(self, feat_ch=1024, img_ch=10):
        super().__init__()

        # compress 1024 -> 128
        self.compress4 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress3 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress2 = ConvGnRelu(feat_ch, 128, 1, 0)
        self.compress1 = ConvGnRelu(feat_ch, 128, 1, 0)

        # progressive fusion
        self.fuse43 = GatedFusion(128)
        self.fuse32 = GatedFusion(128)
        self.fuse21 = GatedFusion(128)

        # bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
        )

        # 64 -> 128
        self.up1_pre = ResidualBlock(256, 192)
        self.skip1_proj = ConvGnRelu(128, 64, 1, 0)
        self.up1_fuse = nn.Sequential(
            ResidualBlock(192 + 64, 128),
            ResidualBlock(128, 128),
        )

        # 128 -> 256
        self.up2_pre = ResidualBlock(128, 96)

        # gated input skip
        self.img_proj = ConvGnRelu(img_ch, 32, 1, 0)
        self.img_gate = nn.Sequential(
            nn.Conv2d(96 + 32, 64, 1), nn.ReLU(True), nn.Conv2d(64, 32, 1), nn.Sigmoid()
        )
        self.up2_fuse = nn.Sequential(
            ResidualBlock(96 + 32, 64),
            ResidualBlock(64, 32),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, feats, img_norm):
        c1, c2, c3, c4 = feats

        f1 = self.compress1(c1)
        f2 = self.compress2(c2)
        f3 = self.compress3(c3)
        f4 = self.compress4(c4)

        x = self.fuse43(f4, f3)
        x = self.fuse32(x, f2)
        x = self.fuse21(x, f1)

        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1_pre(x)

        f1_up = F.interpolate(f1, scale_factor=2, mode="bilinear", align_corners=False)
        f1_up = self.skip1_proj(f1_up)
        x = self.up1_fuse(torch.cat([x, f1_up], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2_pre(x)

        img_feat = self.img_proj(img_norm)
        g = self.img_gate(torch.cat([x, img_feat], dim=1))
        img_feat = img_feat * g
        x = self.up2_fuse(torch.cat([x, img_feat], dim=1))

        x = self.dropout(x)
        return self.final(x)


# ==================== Loss ====================
class PixelWeightedL1Loss(nn.Module):
    def forward(self, pred, target):
        valid = target >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p = pred.squeeze(1)[valid]
        t = target[valid]
        w = 0.2 + 2.8 * torch.sqrt(t.clamp(min=0))
        return (torch.abs(p - t) * w).mean()


# ==================== Full Model ====================
class FullModel(nn.Module):
    def __init__(self, backbone, head, mean, std):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, img):
        x_norm = (img - self.mean) / self.std
        feats = self.backbone(x_norm)
        return self.head(feats, x_norm)
