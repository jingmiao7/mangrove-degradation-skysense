"""Run the trained L5 SkySense++ model on a 10-band 10 m GeoTIFF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input 10-band GeoTIFF")
    parser.add_argument("--output", type=Path, required=True, help="Output degradation-fraction GeoTIFF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned L5 best.pth")
    parser.add_argument("--pretrained-weights", type=Path, required=True,
                        help="SkySense++ Sentinel-2 pretrained backbone checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256,
                        help="Inference stride (fixed at 256 to preserve non-overlapping tile coverage)")
    parser.add_argument("--amp-bfloat16", action="store_true",
                        help="Use BF16 autocast on supported CUDA GPUs")
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser.parse_args()


def pad_patch(patch: np.ndarray, height: int, width: int, patch_size: int) -> np.ndarray:
    """Pad a band-first patch to model size, reflecting valid edge pixels."""
    pad_h, pad_w = patch_size - height, patch_size - width
    if pad_h == 0 and pad_w == 0:
        return patch
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)


def main():
    args = parse_args()
    if args.patch_size != 256:
        raise ValueError("This checkpoint was trained on 256x256 inputs; patch size must be 256.")
    if args.stride != 256:
        raise ValueError("This predictor uses non-overlapping 256x256 tiles; stride must be 256.")
    for path in (args.input, args.checkpoint, args.pretrained_weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    # The backbone loader reads this environment variable during model import/build.
    os.environ["SKYSENSE_PRETRAIN"] = str(args.pretrained_weights.expanduser().resolve())
    from train_l5 import EXPERIMENTS, build_model

    device = torch.device(args.device)
    model, _, _ = build_model(EXPERIMENTS["L5_both_all24_full"], activation_checkpointing=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != "L5_both_all24_full":
        raise ValueError(f"Expected an L5 checkpoint, got {checkpoint.get('experiment')!r}")
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys
               if not ((key.startswith("backbone.") and "lora_" not in key) or key in {"mean", "std"})]
    if unexpected or missing:
        raise RuntimeError(f"Checkpoint/model mismatch. Missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.to(device).eval()
    if args.amp_bfloat16 and device.type != "cuda":
        raise ValueError("BF16 autocast was requested but the selected device is not CUDA.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.input) as src:
        if src.count != 10:
            raise ValueError(f"Expected 10 input bands in documented order, found {src.count}.")
        height, width = src.height, src.width
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=args.nodata,
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=256, blockysize=256)
        with rasterio.open(args.output, "w", **profile) as dst:
            for top in range(0, height, args.stride):
                for left in range(0, width, args.stride):
                    h = min(args.patch_size, height - top)
                    w = min(args.patch_size, width - left)
                    window = Window(left, top, w, h)
                    image = src.read(window=window).astype(np.float32, copy=False)
                    valid = src.dataset_mask(window=window) > 0
                    valid &= np.isfinite(image).all(axis=0)
                    if src.nodata is not None:
                        valid &= ~np.any(image == src.nodata, axis=0)
                    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                    if src.nodata is not None:
                        image[image == src.nodata] = 0.0
                    padded = pad_patch(image, h, w, args.patch_size)
                    tensor = torch.from_numpy(np.ascontiguousarray(padded)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=args.amp_bfloat16):
                            prediction = model(tensor).squeeze(0).squeeze(0)
                    prediction = prediction.float().clamp(0.0, 1.0).cpu().numpy()[:h, :w]
                    prediction[~valid] = args.nodata
                    dst.write(prediction.astype(np.float32), 1, window=window)

    print(f"Wrote prediction: {args.output}")


if __name__ == "__main__":
    main()
