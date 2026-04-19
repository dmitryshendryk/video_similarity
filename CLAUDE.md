# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

S2VS (Self-Supervised Video Similarity) is a PyTorch implementation of the paper "Self-Supervised Video Similarity Learning" (CVPR-W 2023). It trains a ViSiL-based similarity network using self-supervised contrastive learning with multi-type video augmentations for video copy detection and retrieval tasks.

Fork of [gkordo/s2vs](https://github.com/gkordo/s2vs).

## Commands

### Install
```bash
pip install torch torchvision  # install PyTorch separately first
pip install -r requirements.txt
```

### Training
```bash
# Edit scripts/train_ssl.sh to set VIDEO_DATASET and EXPERIMENTS paths, then:
bash scripts/train_ssl.sh
# Uses torchrun with 2 GPUs by default (DistributedDataParallel)
```

### Evaluation
```bash
# With precomputed HDF5 features (preferred):
python evaluation.py --dataset FIVR-200K --dataset_hdf5 <path.hdf5> --model_path <model.pth>

# With raw video files:
python evaluation.py --dataset FIVR-200K --dataset_path <dir> --pattern '{id}/video.*' --model_path <model.pth>

# Omit --model_path to use pretrained s2vs_dns model
```

### Feature Extraction to HDF5
```bash
python extract_features.py --dataset FIVR-200K --dataset_path <dir> --pattern '{id}/video.*' --dataset_hdf5 <output.hdf5>
```

### Frame Extraction for Training Data
```bash
ffmpeg -nostdin -y -vf fps=1 -start_number 0 -q 0 ${video_id}/%05d.jpg -i <path_to_video>
```

## Architecture

### Two-stage pipeline

1. **Feature Extraction** (frozen during training): `model/feature_extractor.py`
   - `VideoNormalizer` → `ResNet50` backbone (`model/extractors/resnet.py`) → region max-pooling → `PCALayer` (LiMAC whitening, pretrained)
   - Output: `(T, R, D)` tensor — T frames, R spatial regions, D=512 dims

2. **Similarity Network** (trainable): `model/similarity_network.py`
   - `ViSiL` class — the only `SimilarityNetwork` variant
   - `index_video()`: applies `Attention` or `Binarization` layer to frame features
   - Frame-to-frame similarity via `ChamferSimilarity` (`model/similarities.py`)
   - `VideoComparator` CNN head refines the frame similarity matrix
   - Video-to-video similarity via second `ChamferSimilarity`

### Training flow (`train.py`)

- `SSLGenerator` (`datasets/generators.py`) creates pairs of weak/strong augmented videos from extracted frames
- `WeakAugmentations`: random crop + horizontal flip
- `StrongAugmentations` (`datasets/augmentations.py`): configurable combination of GT (RandAugment), FT (blur/overlay), TT (temporal shuffle/speed), ViV (video-in-video compositing)
- Transform implementations live in `datasets/transforms/`
- Loss: `InfoNCE + λ*(SSHN_hardneg + SSHN_self) + r*regularization` (`model/losses.py`)
- Optimizer: AdamW with CosineLR scheduler (from `timm`)

### Evaluation flow (`evaluation.py`)

- `EvaluationDataset` enum (`datasets/__init__.py`) dispatches to dataset-specific handlers: `FIVR`, `VCDB`, `EVVE`, `DnS`, `CC_WEB_VIDEO`
- Each dataset class loads annotations from pickle/txt files in `data/`
- `VideoDatasetGenerator` loads raw videos; `HDF5DatasetGenerator` loads precomputed features
- Computes query-vs-database similarity → ranks → mAP

### PyTorch Hub (`hubconf.py`)

Three entry points: `resnet50_LiMAC(dims)`, `s2vs_dns()`, `s2vs_vcdb()`. Pretrained weights auto-downloaded from `mever.iti.gr`.

### Key model classes

| Class | File | Role |
|-------|------|------|
| `FeatureExtractor` (enum) | `model/feature_extractor.py` | Builds normalizer + backbone + PCA |
| `SimilarityNetwork` (enum) | `model/similarity_network.py` | Builds ViSiL model |
| `ViSiL` | `model/similarity_network.py` | Full similarity pipeline |
| `ChamferSimilarity` | `model/similarities.py` | Max-pool → mean aggregation |
| `VideoComparator` | `model/similarities.py` | 4-layer CNN on similarity matrices |
| `Attention` | `model/layers.py` | Context-gated attention for indexing |
| `PCALayer` | `model/layers.py` | Whitening + dim reduction |
| `InfoNCELoss`, `SSHNLoss` | `model/losses.py` | Contrastive + hard negative losses |
| `SSLGenerator` | `datasets/generators.py` | Self-supervised training data loader |

## Evaluation Datasets

Supported via `--dataset` flag: `FIVR-5K`, `FIVR-200K`, `CC_WEB_VIDEO`, `EVVE`, `VCDB`. Feature downloads at `https://mever.iti.gr/s2vs/features/`.

## Notes

- No test suite, linter config, or CI — this is a research codebase
- Python 3 with `.venv` virtualenv present in repo root
- Training requires pre-extracted frames (JPEG at 1fps); evaluation can use raw video or HDF5
- Only the similarity network is trained; the ResNet50 feature extractor stays frozen
- Distributed training uses `torchrun` + `torch.distributed` (configured in `utils.py`)
