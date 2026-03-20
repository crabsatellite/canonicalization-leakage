"""
SMILES invariance test — the core diagnostic for canonicalization leakage.

For each test molecule:
  1. Generate K random SMILES (same molecule, different string)
  2. Recompute SMILES-derived features (Group A) from each random SMILES
  3. Keep graph-derived features (Groups B, C) unchanged
  4. Run model prediction on each variant
  5. Measure prediction variance across variants

If model is truly invariant to SMILES representation, predictions should be identical.
Parallels the NPN circuit invariance test.
"""

import json
import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"

from data_prep import compute_smiles_bigrams_for_strings, generate_random_smiles
from train_mlp import MolMLP, load_config


def run_invariance_test(model_tag="mlp_all", n_test=500, K=30, seed=42):
    """
    Run SMILES invariance test on a trained model.

    Args:
        model_tag: checkpoint name (without _best.pt)
        n_test: number of test molecules to evaluate
        K: number of random SMILES per molecule
        seed: random seed
    """
    config = load_config()
    mc = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"SMILES INVARIANCE TEST")
    print(f"Model: {model_tag}, N_test={n_test}, K={K}")
    print(f"{'='*60}")

    # Load data
    data = np.load(DATA_DIR / "prepared.npz", allow_pickle=True)
    features = data["features"]
    targets = data["targets"]
    test_idx = data["test_idx"]
    smiles = data["smiles"]
    dim_A = int(data["dim_A"])
    dim_B = int(data["dim_B"])
    dim_C = int(data["dim_C"])
    bigram_vocab = list(data["bigram_vocab"])
    feature_mean = data["feature_mean"]
    feature_std = data["feature_std"]

    # Select test subset
    if n_test < len(test_idx):
        subset = np.random.choice(len(test_idx), n_test, replace=False)
        eval_idx = test_idx[subset]
    else:
        eval_idx = test_idx
        n_test = len(eval_idx)

    print(f"Evaluating {n_test} test molecules")

    # Load model
    input_dim = features.shape[1]
    model = MolMLP(input_dim, mc["hidden_dims"], mc["dropout"]).to(device)
    ckpt_path = CKPT_DIR / f"{model_tag}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    # For each test molecule: get predictions under K random SMILES
    all_preds = np.zeros((n_test, K + 1))  # canonical + K random
    true_targets = targets[eval_idx]

    # Canonical predictions
    with torch.no_grad():
        X_canon = torch.tensor(features[eval_idx], dtype=torch.float32).to(device)
        all_preds[:, 0] = model(X_canon).cpu().numpy()

    # Random SMILES predictions
    print(f"Generating {K} random SMILES per molecule...")
    for i, idx in enumerate(eval_idx):
        smi = str(smiles[idx])
        random_smiles = generate_random_smiles(smi, n_random=K)

        # Recompute Group A features from random SMILES
        bigram_feats = compute_smiles_bigrams_for_strings(random_smiles, bigram_vocab)

        # Normalize Group A using training stats
        bigram_feats_norm = (bigram_feats - feature_mean[:dim_A]) / feature_std[:dim_A]

        # Groups B and C stay the same (graph-derived, invariant)
        feat_BC = features[idx, dim_A:]  # (dim_B + dim_C,)
        feat_BC_repeated = np.tile(feat_BC, (K, 1))  # (K, dim_B + dim_C)

        # Concatenate: random Group A + fixed Groups B, C
        feat_variants = np.concatenate([bigram_feats_norm, feat_BC_repeated], axis=1)

        with torch.no_grad():
            X_var = torch.tensor(feat_variants, dtype=torch.float32).to(device)
            preds = model(X_var).cpu().numpy()
            all_preds[i, 1:] = preds

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_test} molecules")

    # Analysis
    print(f"\n--- Results ---")

    # Per-molecule statistics
    pred_means = all_preds.mean(axis=1)
    pred_stds = all_preds.std(axis=1)
    mean_std = pred_stds.mean()
    median_std = np.median(pred_stds)

    # Consistency: fraction of molecules where all K+1 predictions agree
    # (rounded to 2 decimal places, since targets are in Hartrees)
    rounded = np.round(all_preds, 2)
    consistent = np.all(rounded == rounded[:, :1], axis=1)
    consistency_rate = consistent.mean()

    # Canonical performance
    canon_preds = all_preds[:, 0]
    canon_rs, _ = spearmanr(canon_preds, true_targets)
    canon_mae = np.mean(np.abs(canon_preds - true_targets))

    # Orbit-averaged performance
    avg_preds = all_preds.mean(axis=1)
    avg_rs, _ = spearmanr(avg_preds, true_targets)
    avg_mae = np.mean(np.abs(avg_preds - true_targets))

    print(f"Canonical Spearman r_s:     {canon_rs:.4f}")
    print(f"SMILES-averaged Spearman:   {avg_rs:.4f}")
    print(f"Delta (leakage indicator):  {canon_rs - avg_rs:.4f}")
    print(f"")
    print(f"Canonical MAE:              {canon_mae:.4f}")
    print(f"SMILES-averaged MAE:        {avg_mae:.4f}")
    print(f"")
    print(f"Mean prediction std:        {mean_std:.4f}")
    print(f"Median prediction std:      {median_std:.4f}")
    print(f"Consistency (2dp):          {consistency_rate:.1%}")

    # Verdict
    if mean_std < 0.001:
        verdict = "STRONG INVARIANCE"
    elif mean_std < 0.01:
        verdict = "PARTIAL INVARIANCE"
    elif mean_std < 0.05:
        verdict = "WEAK INVARIANCE"
    else:
        verdict = "NOT INVARIANT"
    print(f"\nVerdict: {verdict}")

    # Save results
    results = {
        "model_tag": model_tag,
        "n_test": n_test,
        "K": K,
        "canonical_rs": float(canon_rs),
        "smiles_averaged_rs": float(avg_rs),
        "delta": float(canon_rs - avg_rs),
        "canonical_mae": float(canon_mae),
        "smiles_averaged_mae": float(avg_mae),
        "mean_prediction_std": float(mean_std),
        "median_prediction_std": float(median_std),
        "consistency_2dp": float(consistency_rate),
        "verdict": verdict,
        "per_molecule_stds": pred_stds.tolist(),
    }

    output_path = DATA_DIR / f"invariance_test_{model_tag}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save raw predictions for figure generation
    np.savez_compressed(
        DATA_DIR / f"invariance_preds_{model_tag}.npz",
        all_preds=all_preds,
        true_targets=true_targets,
        eval_idx=eval_idx,
    )

    return results


def main():
    # Test on the full model (all features)
    results_all = run_invariance_test(model_tag="mlp_all", n_test=500, K=30)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Canonical r_s:       {results_all['canonical_rs']:.4f}")
    print(f"SMILES-avg r_s:      {results_all['smiles_averaged_rs']:.4f}")
    print(f"Consistency:         {results_all['consistency_2dp']:.1%}")
    print(f"Verdict:             {results_all['verdict']}")


if __name__ == "__main__":
    main()
