# Canonicalization Leakage: How Canonical Representatives Confound Supervised Learning under Group Symmetry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19097226.svg)](https://doi.org/10.5281/zenodo.19097226)

This repository contains the code, data, and trained models for diagnosing canonicalization leakage in symmetry-reduced datasets. We propose **CL-DIAG**, a diagnostic protocol that detects, localizes, and quantifies how models trained on canonical representatives exploit representation artifacts rather than learning invariant structure.

## Key Results

| Metric | Canonical | NPN-averaged | Note |
|--------|-----------|--------------|------|
| Spearman $r_s$ | 0.788 | 0.254 | 0% prediction consistency |
| After 7× NPN augmentation | 0.786 | **0.777** | 14.2pp above invariant ceiling |

**Signal decomposition** of canonical performance ($r_s = 0.788$):

| Component | $r_s$ | $\Delta$ |
|-----------|-------|----------|
| $S_\text{inv}$: Classical invariant signal | 0.635 | +0.072 |
| $S_\text{neural}$: Neural invariant signal | 0.777 | +0.142 |
| $L_\text{canon}$: Canonicalization leakage | 0.788 | +0.011 |

**Matched-volume control** (all at 2× volume, 862K samples):

| Condition | NPN-avg $r_s$ | Consistency |
|-----------|---------------|-------------|
| NPN augmentation | **0.728** | **38.8%** |
| Noise augmentation | 0.691 | 0.0% |
| Data duplication | 0.172 | 0.0% |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the optimal5 dataset (Goucher, MIT License)
#    See REPRODUCE.md for download instructions

# 3. Prepare features from raw data
cd src && python data_prep.py

# 4. Train baseline MLP on canonical data
python train_mlp.py

# 5. Run NPN invariance test
python npn_invariance_test.py

# 6. Run contamination decomposition
python npn_contamination_decomp.py

# 7. Train with NPN augmentation (0×, 1×, 3×, 7×)
python train_npn_augmented.py

# 8. Run matched-volume control experiment
python augmentation_control.py

# 9. Save figure data and generate paper figures
python save_fig1_predictions.py
python generate_figures.py
```

## Project Structure

```
canonicalization-leakage/
├── src/                    # NPN circuit experiment code
│   ├── data_prep.py        # Feature extraction from raw truth tables
│   ├── compute_measures.py # Boolean function measures (entropy, nonlinearity, etc.)
│   ├── train_mlp.py        # Baseline MLP training
│   ├── npn_invariance_test.py      # NPN invariance testing (CL-DIAG Step 2)
│   ├── npn_contamination_decomp.py # Per-feature decomposition (CL-DIAG Step 3)
│   ├── train_npn_augmented.py      # Augmentation sweep (CL-DIAG Step 4)
│   ├── augmentation_control.py     # Matched-volume control (CL-DIAG Step 5)
│   └── generate_figures.py         # Paper figure generation
├── qm9-smiles/             # QM9 molecular property prediction (Section 5)
│   ├── src/                # SMILES canonicalization experiment code
│   ├── data/               # QM9 experiment results (JSON)
│   └── configs/            # QM9 experiment configuration
├── data/                   # NPN experiment results (JSON)
├── figures/                # Paper figures (PDF)
├── checkpoints/            # Trained model weights
├── configs/                # Experiment configuration
└── REPRODUCE.md            # Step-by-step reproduction guide
```

## Dataset

The NPN5 database (616,126 NPN equivalence classes of 5-input Boolean functions with exact minimum circuit sizes) is from:

- **Source**: [Adam P. Goucher, Optimal Circuits for 5-input Boolean Functions](https://cp4space.hatsya.com/2019/05/28/five-input-boolean-circuits/)
- **License**: MIT
- **Size**: ~16 MB

See `REPRODUCE.md` for download instructions.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU (RTX 3090 used in paper)
- ~1 GB disk space
- Total compute: < 4 GPU-hours

## Citation

```bibtex
@misc{li2026canonicalization,
  author    = {Li, Alex Chengyu},
  title     = {Canonicalization Leakage: How Canonical Representatives
               Confound Supervised Learning under Group Symmetry},
  year      = {2026},
  doi       = {10.5281/zenodo.19097226},
  url       = {https://doi.org/10.5281/zenodo.19097226},
  note      = {Preprint}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Dataset from [Adam P. Goucher](https://cp4space.hatsya.com/) (MIT License).
Circuit complexity values from Donald E. Knuth, *The Art of Computer Programming*.
