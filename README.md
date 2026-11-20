# Hurricane Irma mangrove degradation mapping

This project fine-tunes SkySense++ to map the fraction of mangrove degradation associated with Hurricane Irma from 10-band Sentinel-2 imagery. It includes the model, experiment configuration, exact data splits, and scripts for training and tiled GeoTIFF prediction. The output is a per-pixel degradation fraction between 0 and 1.

## Model configuration

The main model is a SkySense++ ViT-L-style spectral-spatial encoder adapted with LoRA and paired with a progressive gated U-Net decoder. Its exact name in the training script is `skysense_vitl_lora_all_layers_gated_unet`.

| Component | Configuration |
| --- | --- |
| Input | 256 × 256 pixels; 10 Sentinel-2 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| Encoder | SkySense++ ViT-L; patch size 4; embedding dimension 1024; 24 transformer blocks; 16 attention heads; MLP ratio 4 |
| Encoder features | Outputs from blocks 6, 12, 18, and 24 (zero-based indices 5, 11, 17, 23) |
| Initialization | Sentinel-2 pretrained SkySense++ backbone; backbone weights frozen except LoRA adapters |
| Adaptation | LoRA rank 8, alpha 16; applied to attention QKV/output projections and both FFN projections in all 24 blocks |
| Decoder | Progressive gated multiscale U-Net, fusing all four encoder feature outputs and image-level spectral features |
| Output | Per-pixel degradation fraction, clipped to [0, 1] during prediction |
| Normalization | Per-band Z-score normalization using the training-set statistics stored in `src/train_model.py` |

The full recorded run configuration (including split sizes and training settings) is in [`configs/model_configuration.json`](configs/model_configuration.json).

## Reproduction environment

The observed environment is recorded in [`docs/environment_observed.json`](docs/environment_observed.json): Python 3.12.1, PyTorch 2.12.0.dev20260407+cu128, CUDA 12.8. This is a PyTorch nightly build; installing old nightly wheels may require a compatible Python/CUDA image. Do not silently substitute a different build when claiming bitwise reproduction.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The data root must contain `tiles_256/images/*.npy` and `tiles_256/labels/*.npy`. The exact train/validation/test tile manifests are included under `configs/splits/` and contain 244, 52, and 56 tile names respectively.

```bash
export MANGROVE_DATA=/path/to/training_data
export SKYSENSE_PRETRAIN=/path/to/skysense_model_backbone_s2.pth
export OUTPUT_DIR="$PWD/outputs"
python src/train_model.py --seeds 42 --epochs 60 --batch-size 4
```

The recorded run uses seed 42, 60 epochs, batch size 4, BF16 autocast, PyTorch SDPA, deterministic algorithms, zero data-loader workers, and no activation checkpointing. The best validation checkpoint was from epoch 50. The training script defines the AdamW optimizer, weighted L1 loss, cosine schedule, and training-only augmentations.

## Prediction

Input must be a 10-band GeoTIFF in the order listed above and in the same numeric scale as the training arrays. Prediction applies the recorded band normalization and writes a single-band degradation-fraction GeoTIFF, preserving the input grid and setting invalid pixels to -9999.

```bash
python src/predict.py \
  --input /path/to/sentinel2_10band.tif \
  --checkpoint /path/to/main_model/seed_42/best.pth \
  --pretrained-weights /path/to/skysense_model_backbone_s2.pth \
  --output /path/to/degradation_fraction.tif \
  --device cuda --amp-bfloat16
```

Inference uses non-overlapping 256 × 256 windows; edge windows are padded then cropped. Keep patch size and stride at 256. The predictor accepts both current checkpoints and the original checkpoint metadata identifier for backward compatibility.

## Repository map

- `src/train_model.py`: canonical experiment runner; default is the model described above.
- `src/predict.py`: tiled GeoTIFF prediction.
- `src/model_components.py`, `src/standalone_vit.py`, `src/finetune/`: model and backbone implementation.
- `configs/model_configuration.json`: recorded model and run configuration.
- `configs/splits/`: exact train, validation, and test manifests.
- `docs/data_and_weights.md`: data and checkpoint locations, checksums, and provenance.

## Data, weights, and licensing

Training arrays and model weights are intentionally not versioned because of their size and separate redistribution terms. See [`docs/data_and_weights.md`](docs/data_and_weights.md) for expected locations and SHA-256 checksums. No project-level license is assigned yet. Confirm upstream SkySense++ licensing and the rights for all data and checkpoints before making this repository public.
