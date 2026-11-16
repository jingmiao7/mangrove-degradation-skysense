"""
SkySense++ ViT-L backbone (frozen + Attention/FFN LoRA ALL layers)
+ Progressive gated-fusion decoder
+ Gaussian noise augmentation
+ Edge masking augmentation
Output: 256x256 @ 10m
"""
import os, sys, json, time, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from finetune.backbone_loader import build_skysense_backbone

sys.path.insert(0, ".")
sys.path.insert(0, "/home/user/SkySensePlusPlus")

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
        if hasattr(layer, "attn") and hasattr(layer.attn, "qkv") and isinstance(layer.attn.qkv, nn.Linear):
            orig_qkv = layer.attn.qkv
            lora_qkv = LoRALinear(orig_qkv, rank=rank, alpha=alpha)
            layer.attn.qkv = lora_qkv
            lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])

        # ---- attention proj ----
        if hasattr(layer, "attn") and hasattr(layer.attn, "proj") and isinstance(layer.attn.proj, nn.Linear):
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
    print("Attn+FFN LoRA: start_layer=%d (full backbone), rank=%d, %.2fM params" % (start_layer, rank, n / 1e6))
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
            nn.ReLU(True)
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
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch)
        )

        if in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(g, out_ch)
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
            nn.Sigmoid()
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
            nn.Conv2d(96 + 32, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1),
            nn.Sigmoid()
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

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1_pre(x)

        f1_up = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=False)
        f1_up = self.skip1_proj(f1_up)
        x = self.up1_fuse(torch.cat([x, f1_up], dim=1))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
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


# ==================== Year helper ====================
def infer_year_from_item(item):
    for k in ["year", "Year", "img_year"]:
        if k in item:
            try:
                y = int(item[k])
                if y in [2017, 2019, 2021, 2023]:
                    return y
            except Exception:
                pass

    name = item.get("tile_name", "")
    m = re.search(r"(2017|2019|2021|2023)", name)
    if m:
        return int(m.group(1))
    return -1

# ==================== Dataset ====================
class MangroveDataset(Dataset):
    def __init__(self, data_root, ann_file, img_sub, label_sub, augment=False):
        with open(os.path.join(data_root, ann_file)) as f:
            self.annotations = json.load(f)
        self.data_root = data_root
        self.img_sub = img_sub
        self.label_sub = label_sub
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        name = self.annotations[idx]["tile_name"]
        img = np.load(os.path.join(self.data_root, self.img_sub, name + ".npy"))
        label = np.load(os.path.join(self.data_root, self.label_sub, name + ".npy"))
        mp = os.path.join(self.data_root, self.label_sub, name + "_mask.npy")
        mask = np.load(mp) if os.path.exists(mp) else np.ones_like(label)
        label[mask == 0] = -1

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        if self.augment:
            img, label = self._augment(img, label)

        year = infer_year_from_item(self.annotations[idx])
        return img, label, name, year

    def _augment(self, img, label):
        if torch.rand(1).item() < 0.5:
            noise_std = torch.rand(1).item() * 50
            img = img + torch.randn_like(img) * noise_std

        if torch.rand(1).item() < 0.5:
            top = torch.randint(0, 65, (1,)).item()
            bottom = torch.randint(0, 65, (1,)).item()
            left = torch.randint(0, 65, (1,)).item()
            right = torch.randint(0, 65, (1,)).item()
            if top > 0:
                img[:, :top, :] = 0; label[:top, :] = -1
            if bottom > 0:
                img[:, -bottom:, :] = 0; label[-bottom:, :] = -1
            if left > 0:
                img[:, :, :left] = 0; label[:, :left] = -1
            if right > 0:
                img[:, :, -right:] = 0; label[:, -right:] = -1

        if torch.rand(1).item() < 0.5:
            img = img.flip(1); label = label.flip(0)
        if torch.rand(1).item() < 0.5:
            img = img.flip(2); label = label.flip(1)

        if torch.rand(1).item() < 0.2:
            n_drop = torch.randint(1, 3, (1,)).item()
            bands = torch.randperm(10)[:n_drop]
            img[bands] = 0

        return img, label


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


# ==================== Main ====================
def main():
    n_gpu = torch.cuda.device_count()
    print("GPUs: %d" % n_gpu)

    work_dir = "/home/user//home/user"
    os.makedirs(work_dir, exist_ok=True)
    log_file = open(os.path.join(work_dir, "train.log"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    bb = build_skysense_backbone("/home/user/SkySensePlusPlus/pretrain/skysense_model_backbone_s2.pth")

    # Apply LoRA to ALL layers (start_layer=0)
    lora_params = apply_attn_ffn_lora(bb, rank=8, alpha=16, start_layer=0)

    head = UNetGatedDecoder()
    n_head = sum(p.numel() for p in head.parameters())
    n_lora = sum(p.numel() for p in lora_params)
    log("LoRA: %.2fM | Gated Head: %.2fM | Total trainable: %.2fM" % (
        n_lora/1e6, n_head/1e6, (n_lora+n_head)/1e6))

    mean_t = torch.tensor([172.6143,304.1498,203.0544,486.6588,1492.7885,
                           1896.2063,1988.8759,2119.2815,653.4074,281.6476]).view(1,10,1,1)
    std_t = torch.tensor([79.7107,120.3797,132.46,191.0323,730.7147,
                          945.5046,1017.748,1060.7884,409.2817,223.9724]).view(1,10,1,1)

    model = FullModel(bb, head, mean_t, std_t)
    if n_gpu > 1:
        model = nn.DataParallel(model)
        log("DataParallel: %d GPUs" % n_gpu)

    model = model.cuda()
    batch_size = 4 * n_gpu
    if batch_size == 0:
        raise RuntimeError("No CUDA GPU found.")

    optimizer = AdamW([
        {"params": lora_params, "lr": 5e-4, "weight_decay": 0.01},
        {"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    criterion = PixelWeightedL1Loss()

    train_ds = MangroveDataset("/home/user/SkySensePlusPlus/training_data/", "metadata/train_256_all.json",
                               "tiles_256/images", "tiles_256/labels", augment=True)
    val_ds = MangroveDataset("/home/user/SkySensePlusPlus/training_data/", "metadata/val_256_all.json",
                             "tiles_256/images", "tiles_256/labels", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    log("\n=== SkySense++ Attn+FFN LoRA (ALL layers) + Gated Decoder 10m ===")
    log("  Decoder: progressive gated fusion")
    log("  Augment: gaussian noise + edge mask + flip + band dropout")
    log("  batch=%d, 60 epochs" % batch_size)
    log("  Train: %d, Val: %d\n" % (len(train_ds), len(val_ds)))

    best_r2 = -999
    for epoch in range(1, 61):
        model.train()
        train_loss, n_batch = 0, 0
        t0 = time.time()

        for img, label, _, _ in train_loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1

        scheduler.step()

        model.eval()
        preds_all, trues_all = [], []
        year_preds = {2017: [], 2019: [], 2021: [], 2023: []}
        year_trues = {2017: [], 2019: [], 2021: [], 2023: []}

        with torch.no_grad():
            for img, label, name, year in val_loader:
                img, label = img.cuda(), label.cuda()
                pred = model(img)
                valid = label >= 0
                if valid.sum() > 0:
                    p = pred.squeeze(1)[valid].cpu().numpy()
                    t = label[valid].cpu().numpy()
                    preds_all.append(p)
                    trues_all.append(t)

                    y = int(year[0]) if torch.is_tensor(year) else int(year)
                    if y in year_preds:
                        year_preds[y].append(p)
                        year_trues[y].append(t)

        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)
        mae = np.abs(preds_all - trues_all).mean()
        rmse = np.sqrt(((preds_all - trues_all)**2).mean())
        r2 = 1 - ((trues_all - preds_all)**2).sum() / max(((trues_all - trues_all.mean())**2).sum(), 1e-8)

        year_metrics = {}
        for yy in [2017, 2019, 2021, 2023]:
            if len(year_preds[yy]) > 0:
                yp = np.concatenate(year_preds[yy])
                yt = np.concatenate(year_trues[yy])
                y_mae = np.abs(yp - yt).mean()
                y_rmse = np.sqrt(((yp - yt) ** 2).mean())
                y_r2 = 1 - ((yt - yp) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-8)
            else:
                y_mae, y_rmse, y_r2 = np.nan, np.nan, np.nan

            year_metrics[str(yy)] = {
                "r2": float(y_r2) if np.isfinite(y_r2) else None,
                "mae": float(y_mae) if np.isfinite(y_mae) else None,
                "rmse": float(y_rmse) if np.isfinite(y_rmse) else None,
            }

        elapsed = time.time() - t0

        log("Ep %02d | loss=%.4f | R2=%.4f MAE=%.4f RMSE=%.4f | 2017_R2=%.4f 2019_R2=%.4f 2021_R2=%.4f 2023_R2=%.4f | %.0fs" % (
            epoch, train_loss/max(n_batch,1), r2, mae, rmse,
            year_metrics['2017']['r2'] if year_metrics['2017']['r2'] is not None else float('nan'),
            year_metrics['2019']['r2'] if year_metrics['2019']['r2'] is not None else float('nan'),
            year_metrics['2021']['r2'] if year_metrics['2021']['r2'] is not None else float('nan'),
            year_metrics['2023']['r2'] if year_metrics['2023']['r2'] is not None else float('nan'),
            elapsed))

        real_head = model.module.head if hasattr(model, 'module') else model.head
        real_bb = model.module.backbone if hasattr(model, 'module') else model.backbone

        torch.save({
            "epoch": epoch,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "year_metrics": year_metrics,
            "head_state": real_head.state_dict(),
            "lora_state": {n: p.data.cpu() for n, p in real_bb.named_parameters()
                           if ("lora_A" in n or "lora_B" in n)},
        }, os.path.join(work_dir, "last.pth"))

        if r2 > best_r2:
            best_r2 = r2
            torch.save({
                "epoch": epoch,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "year_metrics": year_metrics,
                "head_state": real_head.state_dict(),
                "lora_state": {n: p.data.cpu() for n, p in real_bb.named_parameters()
                               if ("lora_A" in n or "lora_B" in n)},
            }, os.path.join(work_dir, "best.pth"))
            log("  -> Best R2=%.4f" % r2)

        with open(os.path.join(work_dir, "metrics_history.jsonl"), "a") as mf:
            mf.write(json.dumps({
                "epoch": epoch,
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "year_metrics": year_metrics
            }) + "\n")

    log("\n=== Done! Best R2=%.4f ===" % best_r2)
    log_file.close()

if __name__ == "__main__":
    main()