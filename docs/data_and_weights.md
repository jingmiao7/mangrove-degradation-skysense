# Data and model artifacts

This repository contains code and experiment metadata for fine-tuning SkySense++ to map mangrove degradation fraction associated with Hurricane Irma. The training arrays and model checkpoints are stored separately and are not included in the repository.

## Training data

- Required layout: `tiles_256/images/<tile_name>.npy`, `tiles_256/labels/<tile_name>.npy`; optional label validity mask: `tiles_256/labels/<tile_name>_mask.npy`.
- The split lists in `configs/splits/` contain 244 training, 52 validation, and 56 test tile names (352 total). They are the exact manifests used by the logged experiment.
- The data files are not bundled. Use the project data source and follow its access and attribution terms.

## Weights

| Artifact | Filename | Size | SHA-256 |
| --- | --- | ---: | --- |
| SkySense++ Sentinel-2 pretrained backbone | `skysense_model_backbone_s2.pth` | 1,211,050,209 bytes | `f0dd678f8e324f15f146e867ac914c640c868242baa0516ddb21e0e0b69d0315` |
| Fine-tuned model checkpoint, seed 42 | `best.pth` | 35,183,829 bytes | `fab2b181be1c29d83fb2ee98d27cabd3d2b1cc16ac9b3402d5a951cbe98e9ae4` |

Place the pretrained weights at `weights/skysense_model_backbone_s2.pth`, or set `SKYSENSE_PRETRAIN` to their location. Pass the fine-tuned checkpoint path to prediction with `--checkpoint`. Verify files with `sha256sum <file>`.

The fine-tuned checkpoint stores the learned/trainable model state and experiment metadata, not the full frozen pretrained backbone. Prediction therefore needs both the fine-tuned checkpoint and the exact compatible pretrained backbone file.

## Source and licensing

The model implementation and training workflow are based on SkySense++ and the associated experiment code. The repository does not include the training arrays or checkpoints and does not grant rights to those artifacts. Consult their original sources for attribution, access, and licensing terms.
