"""
Master pipeline runner for qm9-smiles canonicalization leakage.
Runs CL-DIAG end-to-end on QM9 molecular property prediction.

Steps:
  1. Data preparation (download QM9, compute features)
  2. Baseline training (canonical SMILES, all feature groups)
  3. SMILES invariance test (core diagnostic)
  4. Contamination decomposition (per-feature-group)
  5. Augmentation sweep (1x, 3x, 7x)
  6. Matched-volume control (SMILES aug vs duplication vs noise)

Usage:
  python run_pipeline.py              # Run all steps
  python run_pipeline.py --step 1     # Run specific step
  python run_pipeline.py --step 1,2,3 # Run specific steps
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step1_data_prep():
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    from data_prep import main as data_main
    data_main()


def step2_baseline():
    print("\n" + "="*70)
    print("STEP 2: BASELINE TRAINING (canonical SMILES)")
    print("="*70)
    from train_mlp import main as train_main
    train_main()


def step3_invariance():
    print("\n" + "="*70)
    print("STEP 3: SMILES INVARIANCE TEST")
    print("="*70)
    from smiles_invariance_test import main as inv_main
    inv_main()


def step4_contamination():
    print("\n" + "="*70)
    print("STEP 4: CONTAMINATION SOURCE DECOMPOSITION")
    print("="*70)
    from contamination_decomp import main as contam_main
    contam_main()


def step5_augmentation():
    print("\n" + "="*70)
    print("STEP 5: AUGMENTATION SWEEP")
    print("="*70)
    from train_augmented import main as aug_main
    aug_main()


def step6_control():
    print("\n" + "="*70)
    print("STEP 6: MATCHED-VOLUME CONTROL")
    print("="*70)
    from augmentation_control import main as control_main
    control_main()


def step7_augmented_invariance():
    print("\n" + "="*70)
    print("STEP 7: AUGMENTED MODEL INVARIANCE TEST")
    print("="*70)
    from test_augmented_invariance import main as aug_inv_main
    aug_inv_main()


STEPS = {
    1: ("Data preparation", step1_data_prep),
    2: ("Baseline training", step2_baseline),
    3: ("SMILES invariance test", step3_invariance),
    4: ("Contamination decomposition", step4_contamination),
    5: ("Augmentation sweep", step5_augmentation),
    6: ("Matched-volume control", step6_control),
    7: ("Augmented model invariance test", step7_augmented_invariance),
}


def main():
    parser = argparse.ArgumentParser(description="CL-DIAG pipeline for QM9")
    parser.add_argument("--step", type=str, default=None,
                        help="Comma-separated step numbers (e.g., '1,2,3'). Default: all.")
    args = parser.parse_args()

    if args.step:
        steps = [int(s.strip()) for s in args.step.split(",")]
    else:
        steps = list(STEPS.keys())

    print("CL-DIAG Pipeline for QM9 SMILES Canonicalization Leakage")
    print(f"Running steps: {steps}")

    for s in steps:
        name, func = STEPS[s]
        print(f"\n>>> Step {s}: {name}")
        func()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
