# Mangrove degradation mapping: reproducibility package

This repository packages the L5 experiment used to predict mangrove degradation fraction from 10-band Sentinel-2 imagery, together with exact train/validation/test tile manifests and a tiled GeoTIFF prediction entry point.

## Contents and exclusions

Code, configuration, split manifests, and documentation are versioned. Training arrays, pretrained SkySense++ backbone, fine-tuned checkpoint, and prediction rasters are deliberately excluded because of their size and separate data/model redistribution terms. See [`docs/data_and_weights.md`](docs/data_and_weights.md) for expected layout, original local paths, sizes, and SHA-256 checksums.

## Reproduction environment

The observed environment is recorded in [`docs/environment_observed.json`](docs/environment_observed.json): Python 3.12.1, PyTorch 2.12.0.dev20260407+cu128, CUDA 12.8. This is a PyTorch nightly build; installing old nightly wheels may require a compatible Python/CUDA image. Do not silently substitute a different build when claiming bitwise reproduction. Install with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The data root must contain `tiles_256/images/*.npy` and `tiles_256/labels/*.npy`. The exact split manifests are included under `configs/splits/` and reference tile names in those directories.

```bash
export MANGROVE_DATA=/path/to/training_data
export SKYSENSE_PRETRAIN=/path/to/skysense_model_backbone_s2.pth
export OUTPUT_DIR="$PWD/outputs"
python src/train_l5.py --experiments L5_both_all24_full --seeds 42 \
  --epochs 60 --batch-size 4
```

The main run uses 24 ViT-L blocks; LoRA rank 8, alpha 16 on attention and FFN projections in all blocks; gated multiscale U-Net decoder; four training-only augmentations; 60 epochs; batch size 4; seed 42; BF16 autocast; PyTorch SDPA; zero data-loader workers; deterministic algorithms; no activation checkpointing. The saved best checkpoint was epoch 50. Optimizer, loss, and scheduler parameters are defined in the canonical training script. Training writes checkpoints and metrics under `OUTPUT_DIR`.

## Prediction

Input must be a 10-band GeoTIFF in the training band order and numeric scale. The predictor applies recorded per-band Z-score normalization and writes a single-band degradation-fraction GeoTIFF in [0, 1], retaining the input grid and setting invalid pixels to -9999.

```bash
python src/predict.py \
  --input /path/to/sentinel2_10band.tif \
  --checkpoint /path/to/L5_both_all24_full/seed_42/best.pth \
  --pretrained-weights /path/to/skysense_model_backbone_s2.pth \
  --output /path/to/degradation_fraction.tif \
  --device cuda --amp-bfloat16
```

Inference uses non-overlapping 256×256 windows; edge windows are padded then cropped. Keep patch size and stride at 256.

## Repository map

- `src/train_l5.py`: canonical experiment runner (default experiment L5).
- `src/predict.py`: tiled prediction using the L5 checkpoint format.
- `src/train_full_lora.py`, `src/standalone_vit.py`, `src/finetune/`: model/backbone implementation required by the runner.
- `configs/main_l5_seed42_observed.json`: recorded main-run settings and split sizes.
- `configs/splits/`: exact train, validation, and test manifests.
- `docs/`: observed environment, data and weight provenance, limitations.

## Licensing and scope

No project-level license is assigned yet. This repository includes adapted SkySense++ implementation files. Confirm upstream licensing/redistribution terms and rights for data and checkpoints before making it public. Reproduction requires matching the documented data, weights, preprocessing, and runtime versions.
