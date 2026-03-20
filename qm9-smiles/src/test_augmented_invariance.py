"""
Run SMILES invariance test on augmented models to measure consistency improvement.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smiles_invariance_test import run_invariance_test
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    results = {}

    # Test baseline (0x)
    print("Testing baseline model (0x augmentation)...")
    r = run_invariance_test(model_tag="mlp_all", n_test=500, K=30)
    results["0x"] = r

    # Test augmented models
    for k in [1, 3, 7]:
        tag = f"mlp_{k}x_augment"
        print(f"\nTesting {k}x augmented model...")
        r = run_invariance_test(model_tag=tag, n_test=500, K=30)
        results[f"{k}x"] = r

    # Save combined results
    output_path = DATA_DIR / "augmented_invariance_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("AUGMENTATION INVARIANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<10} {'Canon r_s':>10} {'Avg r_s':>10} {'Delta':>8} {'Std':>8} {'Consist':>8}")
    print(f"{'-'*54}")
    for name in ["0x", "1x", "3x", "7x"]:
        r = results[name]
        print(f"{name:<10} {r['canonical_rs']:>10.4f} {r['smiles_averaged_rs']:>10.4f} "
              f"{r['delta']:>8.4f} {r['mean_prediction_std']:>8.4f} {r['consistency_2dp']:>7.1%}")


if __name__ == "__main__":
    main()
