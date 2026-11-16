# Data and model artifacts

The binary artifacts below are not stored in this Git repository. Obtain them from the original project storage and verify checksums before training or prediction.

## Training data

- Original local root: `/home/user/SkySensePlusPlus/training_data`
- Required layout: `tiles_256/images/<tile_name>.npy`, `tiles_256/labels/<tile_name>.npy`; optional label validity mask: `tiles_256/labels/<tile_name>_mask.npy`.
- The split lists in `configs/splits/` contain 244 training, 52 validation, and 56 test tile names (352 total). They are the exact manifests used by the logged experiment.
- The data files themselves are not bundled. Ensure you have the right to use/share the NAIP/Sentinel-derived arrays before publishing them.

## Weights

| Artifact | Original local path | Size | SHA-256 |
| --- | --- | ---: | --- |
| SkySense++ Sentinel-2 pretrained backbone | `/home/user/SkySensePlusPlus/pretrain/skysense_model_backbone_s2.pth` | 1,211,050,209 bytes | `f0dd678f8e324f15f146e867ac914c640c868242baa0516ddb21e0e0b69d0315` |
| Fine-tuned main-model checkpoint, seed 42 | `/home/user/ablation_three_groups/L5_both_all24_full/seed_42/best.pth` | 35,183,829 bytes | `fab2b181be1c29d83fb2ee98d27cabd3d2b1cc16ac9b3402d5a951cbe98e9ae4` |

Copy/symlink the pretrained weights to `weights/skysense_model_backbone_s2.pth`, or provide `SKYSENSE_PRETRAIN`. The fine-tuned checkpoint can be kept outside the repository and passed with `--checkpoint` at prediction time. Verify locally with `sha256sum <file>`.

The fine-tuned checkpoint stores the learned/trainable model state and experiment metadata, not the full frozen pretrained backbone. Prediction therefore needs both the fine-tuned checkpoint and the exact compatible pretrained backbone file.

## Source provenance / publication caution

Training/model files were copied from the local SkySense++ project and the experiment runner. No upstream license or checkpoint redistribution notice was found in the inspected project tree. Resolve those terms and cite the original model/data sources before making this repository public; this repository does not grant rights to those artifacts.
