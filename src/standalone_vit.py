"""
Standalone ViT-L backbone - no mmcv/mmengine dependency.
Compatible with SkySense++ pretrained weights and LoRA training code.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=(256, 256), patch_size=4, in_channels=10, embed_dims=1024
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.num_patches = (img_size[0] // self.patch_size[0]) * (
            img_size[1] // self.patch_size[1]
        )
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class FFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, drop_rate=0.0):
        super().__init__()
        # Structure matches mmseg: layers[0][0]=Linear, layers[1]=Linear
        # This is critical for LoRA compatibility
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(embed_dims, feedforward_channels),
                        nn.GELU(),
                        nn.Dropout(drop_rate),
                    ]
                ),
                nn.Linear(feedforward_channels, embed_dims),
            ]
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, identity=None):
        out = x
        for m in self.layers[0]:
            out = m(out)
        out = self.layers[1](out)
        out = self.dropout(out)
        return (identity if identity is not None else x) + out


class MultiheadAttention(nn.Module):
    def __init__(
        self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, qkv_bias=True
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        with_cp=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.attn = MultiheadAttention(
            embed_dims,
            num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.ffn = FFN(embed_dims, feedforward_channels, drop_rate=drop_rate)
        self.with_cp = with_cp

    def forward(self, x):
        def _inner(x):
            x = x + self.attn(self.norm1(x))
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner, x, use_reentrant=False)
        return _inner(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(256, 256),
        patch_size=4,
        in_channels=10,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(5, 11, 17, 23),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=False,
        with_cp=True,
        norm_eval=True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.out_indices = out_indices
        self.with_cls_token = with_cls_token
        self.norm_eval = norm_eval

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dims)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        feedforward_channels = int(embed_dims * mlp_ratio)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    qkv_bias=qkv_bias,
                    with_cp=with_cp,
                )
                for i in range(num_layers)
            ]
        )

        for idx in out_indices:
            setattr(self, f"norm{idx}", nn.LayerNorm(embed_dims, eps=1e-6))

    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embed(x)
        pos = self.pos_embed[:, 1:, :]
        if pos.shape[1] != x.shape[1]:
            s = int(pos.shape[1] ** 0.5)
            pos = pos.reshape(1, s, s, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(H, W), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        x = x + pos
        x = self.drop_after_pos(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)
