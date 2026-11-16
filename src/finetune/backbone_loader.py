import torch
import torch.nn.functional as F
from standalone_vit import VisionTransformer


def _resize_pos_embed(pe, target_shape):
    if pe.ndim != 3 or pe.shape[1] <= 1 or tuple(pe.shape) == tuple(target_shape):
        return pe
    cls_tok = pe[:, :1, :]
    pat = pe[:, 1:, :]
    old_s = int(pat.shape[1] ** 0.5)
    new_tokens = target_shape[1] - 1
    new_s = int(new_tokens**0.5)
    if old_s * old_s != pat.shape[1] or new_s * new_s != new_tokens:
        return pe
    pat = pat.reshape(1, old_s, old_s, pe.shape[-1]).permute(0, 3, 1, 2)
    pat = F.interpolate(pat, size=(new_s, new_s), mode="bicubic", align_corners=False)
    pat = pat.permute(0, 2, 3, 1).reshape(1, new_s * new_s, pe.shape[-1])
    return torch.cat([cls_tok, pat], dim=1)


def _remap_key(k):
    # patch embedding
    k = k.replace("patch_embed.projection.", "patch_embed.projection.")
    # norms
    k = k.replace(".ln1.", ".norm1.")
    k = k.replace(".ln2.", ".norm2.")
    # attention
    k = k.replace(".attn.attn.in_proj_weight", ".attn.qkv.weight")
    k = k.replace(".attn.attn.in_proj_bias", ".attn.qkv.bias")
    k = k.replace(".attn.attn.out_proj.weight", ".attn.proj.weight")
    k = k.replace(".attn.attn.out_proj.bias", ".attn.proj.bias")
    return k


def build_skysense_backbone(pretrain_path):
    bb = VisionTransformer(
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
    )

    ckpt = torch.load(pretrain_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict")

    model_state = bb.state_dict()
    mapped = {}

    for k, v in ckpt.items():
        nk = _remap_key(k)
        if nk in model_state:
            mapped[nk] = v

    if "pos_embed" in mapped and "pos_embed" in model_state:
        mapped["pos_embed"] = _resize_pos_embed(
            mapped["pos_embed"], model_state["pos_embed"].shape
        )

    msg = bb.load_state_dict(mapped, strict=False)

    print(f"Backbone loaded from: {pretrain_path}")
    print(f"Matched keys: {len(mapped)}/{len(model_state)}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    for p in bb.parameters():
        p.requires_grad = False

    return bb
