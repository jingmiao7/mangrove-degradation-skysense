"""
Pure UNet v2
Input: 10-band 256x256
Output: 1-band 256x256
"""
import os, sys, json, time, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, ".")

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
            nn.ReLU(True)
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


# ==================== Pure UNet ====================
class PureUNetV2(nn.Module):
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
        e1 = self.enc1(x)                  # 256
        e2 = self.enc2(self.pool(e1))      # 128
        e3 = self.enc3(self.pool(e2))      # 64
        e4 = self.enc4(self.pool(e3))      # 32

        b = self.bottleneck(self.pool(e4)) # 16

        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d1 = self.dropout(d1)
        return self.final(d1)


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


# ==================== Main ====================
def main():
    n_gpu = torch.cuda.device_count()
    print("GPUs: %d" % n_gpu)

    work_dir = "work_dirs/pure_unet_v2"
    os.makedirs(work_dir, exist_ok=True)
    log_file = open(os.path.join(work_dir, "train.log"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    model = PureUNetV2(in_ch=10)

    if n_gpu > 1:
        model = nn.DataParallel(model)
        log("DataParallel: %d GPUs" % n_gpu)

    model = model.cuda()
    batch_size = 4 * n_gpu
    if batch_size == 0:
        raise RuntimeError("No CUDA GPU found.")

    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Pure UNet trainable: %.2fM" % (n_model / 1e6))

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    criterion = PixelWeightedL1Loss()

    train_ds = MangroveDataset(
        "training_data/", "metadata/train_256_all.json",
        "tiles_256/images", "tiles_256/labels", augment=True
    )
    val_ds = MangroveDataset(
        "training_data/", "metadata/val_256_all.json",
        "tiles_256/images", "tiles_256/labels", augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    log("\n=== Pure UNet v2 ===")
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
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
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
                    p_arr = pred.squeeze(1)[valid].cpu().numpy()
                    t_arr = label[valid].cpu().numpy()
                    preds_all.append(p_arr)
                    trues_all.append(t_arr)

                    y = int(year[0]) if torch.is_tensor(year) else int(year)
                    if y in year_preds:
                        year_preds[y].append(p_arr)
                        year_trues[y].append(t_arr)

        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)
        mae = np.abs(preds_all - trues_all).mean()
        rmse = np.sqrt(((preds_all - trues_all) ** 2).mean())
        r2 = 1 - ((trues_all - preds_all) ** 2).sum() / max(((trues_all - trues_all.mean()) ** 2).sum(), 1e-8)

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
            epoch, train_loss / max(n_batch, 1), r2, mae, rmse,
            year_metrics['2017']['r2'] if year_metrics['2017']['r2'] is not None else float('nan'),
            year_metrics['2019']['r2'] if year_metrics['2019']['r2'] is not None else float('nan'),
            year_metrics['2021']['r2'] if year_metrics['2021']['r2'] is not None else float('nan'),
            year_metrics['2023']['r2'] if year_metrics['2023']['r2'] is not None else float('nan'),
            elapsed))

        real_model = model.module if hasattr(model, 'module') else model
        torch.save({
            "epoch": epoch,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
                "year_metrics": year_metrics,
            "model_state": real_model.state_dict(),
        }, os.path.join(work_dir, "last.pth"))

        if r2 > best_r2:
            best_r2 = r2
            torch.save({
                "epoch": epoch,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "year_metrics": year_metrics,
                "model_state": real_model.state_dict(),
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
