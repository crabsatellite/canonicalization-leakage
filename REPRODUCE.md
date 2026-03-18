# Reproduction Guide

Step-by-step instructions to reproduce all results from the paper.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 3090 used in paper)
- ~1 GB disk space
- ~4 GPU-hours total compute time

```bash
pip install -r requirements.txt
```

## Stage 1: Data Preparation (~5 minutes)

### Download the optimal5 dataset

The NPN5 database by Adam P. Goucher is available under MIT License:

```bash
# Download from the original source
git clone https://gitlab.com/apgoucher/optimal5.git data/optimal5
```

### Prepare features

```bash
cd src
python data_prep.py
```

This extracts truth table bits, Fourier spectrum, and 10 measures for all 616,126 NPN classes. Output: `data/prepared.npz`.

Verify: `python -c "import numpy as np; d = np.load('../data/prepared.npz'); print(f'Samples: {len(d[\"targets\"])}')"` should print `Samples: 616126`.

## Stage 2: Baseline Training (~4 minutes)

```bash
python train_mlp.py
```

Expected: Spearman $r_s \approx 0.765$ on the canonical test set. Checkpoint saved to `checkpoints/`.

## Stage 3: NPN Invariance Test (~5 minutes)

```bash
python npn_invariance_test.py
```

Expected: 0% prediction consistency across NPN transforms. Canonical $r_s \approx 0.788$ on the 500-function evaluation subset drops to $\approx 0.254$ when NPN-averaged.

## Stage 4: Contamination Decomposition (~15 minutes)

```bash
python npn_contamination_decomp.py
```

Expected: Invariant measures show zero contamination (std = 0.000, consistency = 100%). Truth table bits are most contaminated (std $\approx 1.66$).

## Stage 5: NPN Augmentation Sweep (~30 minutes)

```bash
python train_npn_augmented.py
```

Trains four models (0×, 1×, 3×, 7× augmentation). Expected: NPN-averaged $r_s$ increases monotonically from 0.254 to 0.777.

## Stage 6: Matched-Volume Control (~10 minutes)

```bash
python augmentation_control.py
```

Expected: NPN augmentation ($r_s \approx 0.728$) strictly outperforms data duplication ($\approx 0.172$) and noise augmentation ($\approx 0.691$) at equal volume.

## Stage 7: Generate Figures

```bash
python generate_figures.py
```

Produces 5 PDF figures in `figures/`.

## Validation Criteria

Results are valid if:
1. NPN invariance test shows 0% consistency on the baseline model
2. Invariant measures show zero NPN prediction std (0.000)
3. NPN-averaged $r_s$ increases monotonically with augmentation level
4. NPN augmentation outperforms both duplication and noise at matched volume
5. The 7× augmented model achieves NPN-averaged $r_s > 0.635$ (the classical invariant ceiling)

## Hardware Notes

- **GPU**: Any CUDA-capable GPU with 8+ GB VRAM. Paper uses RTX 3090 24GB.
- **Training times**: Scale linearly with augmentation level. 7× takes ~29 min on RTX 3090.
- **Reproducibility**: Random seed is fixed. Results should match within rounding precision.

## Pre-computed Results

All experiment results are provided in `data/` as JSON files. Trained model checkpoints are in `checkpoints/`. These can be used to verify results without re-running experiments.
