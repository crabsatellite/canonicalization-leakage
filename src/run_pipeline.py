"""
Run the full NPN5 experimental pipeline.

Usage:
    python run_pipeline.py              # Run all steps
    python run_pipeline.py --step 3     # Run from step 3
    python run_pipeline.py --step 5 6   # Run steps 5 and 6 only
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

STEPS = [
    (1, "data_prep.py",             "Data preparation"),
    (2, "train_npn_augmented.py",   "Baseline + augmentation sweep (0x, 1x, 3x, 7x)"),
    (3, "npn_contamination_decomp.py", "Per-feature contamination decomposition"),
    (4, "augmentation_control.py",  "Matched-volume control experiments"),
    (5, "multi_seed_run.py",        "Multi-seed variance estimation"),
    (6, "save_fig1_predictions.py", "Save Figure 1 prediction data"),
    (7, "generate_figures.py",      "Generate all figures"),
]


def run_step(num, script, desc):
    print(f"\n{'='*60}")
    print(f"Step {num}: {desc}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"\nStep {num} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="NPN5 experimental pipeline")
    parser.add_argument("--step", type=int, nargs="*",
                        help="Run specific step(s). Without args: run all.")
    args = parser.parse_args()

    if args.step:
        for num, script, desc in STEPS:
            if num in args.step:
                run_step(num, script, desc)
    else:
        for num, script, desc in STEPS:
            run_step(num, script, desc)

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
