"""
Contamination source decomposition for SMILES features.

Trains separate models on each feature group and runs the SMILES invariance test
on each to identify which features carry canonical SMILES bias.

Expected hierarchy (paralleling NPN circuit results):
  - Group A (SMILES bigrams): HIGH contamination — changes with SMILES randomization
  - Group B (Morgan FP): ZERO contamination — graph-derived, invariant
  - Group C (Descriptors): ZERO contamination — graph-derived, invariant
  - Groups B+C: ZERO contamination — all invariant
"""

import json
import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from data_prep import compute_smiles_bigrams_for_strings, generate_random_smiles
from train_mlp import MolMLP, load_config, run_training

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"


def invariance_test_for_group(model, features, targets, smiles, eval_idx,
                              dim_A, bigram_vocab, feature_mean, feature_std,
                              feature_slice, group_name, K=30, device='cpu'):
    """
    Run invariance test for a specific feature group model.

    For Group A models: recompute bigram features from random SMILES
    For Group B/C models: features don't change → predictions should be identical
    """
    n_test = len(eval_idx)
    model.eval()

    # Determine if this group includes Group A (non-invariant)
    start, end = feature_slice
    includes_A = start < dim_A  # Group A occupies [0, dim_A)

    all_preds = np.zeros((n_test, K + 1))
    true_targets = targets[eval_idx]

    # Canonical predictions
    feat_subset = features[eval_idx, start:end]
    with torch.no_grad():
        X = torch.tensor(feat_subset, dtype=torch.float32).to(device)
        all_preds[:, 0] = model(X).cpu().numpy()

    if not includes_A:
        # Invariant features only — all random SMILES give identical features
        # So predictions are trivially identical
        all_preds[:, 1:] = all_preds[:, 0:1]
    else:
        # Group includes non-invariant SMILES features
        for i, idx in enumerate(eval_idx):
            smi = str(smiles[idx])
            random_smiles = generate_random_smiles(smi, n_random=K)

            # Recompute bigram features
            bigram_feats = compute_smiles_bigrams_for_strings(random_smiles, bigram_vocab)
            bigram_feats_norm = (bigram_feats - feature_mean[:dim_A]) / feature_std[:dim_A]

            if end <= dim_A:
                # Pure Group A
                feat_variants = bigram_feats_norm[:, start:end]
            else:
                # A + something else
                feat_A_part = bigram_feats_norm[:, start:min(end, dim_A)]
                feat_rest = features[idx, max(start, dim_A):end]
                feat_rest_rep = np.tile(feat_rest, (K, 1))
                feat_variants = np.concatenate([feat_A_part, feat_rest_rep], axis=1)

            with torch.no_grad():
                X_var = torch.tensor(feat_variants, dtype=torch.float32).to(device)
                all_preds[i, 1:] = model(X_var).cpu().numpy()

    # Compute metrics
    pred_stds = all_preds.std(axis=1)
    mean_std = pred_stds.mean()

    rounded = np.round(all_preds, 2)
    consistent = np.all(rounded == rounded[:, :1], axis=1)
    consistency_rate = consistent.mean()

    canon_rs, _ = spearmanr(all_preds[:, 0], true_targets)
    avg_preds = all_preds.mean(axis=1)
    avg_rs, _ = spearmanr(avg_preds, true_targets)

    return {
        "group": group_name,
        "canonical_rs": float(canon_rs),
        "smiles_averaged_rs": float(avg_rs),
        "mean_std": float(mean_std),
        "consistency": float(consistency_rate),
        "delta": float(canon_rs - avg_rs),
    }


def main():
    config = load_config()
    mc = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)

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

    dim_total = dim_A + dim_B + dim_C

    groups = [
        ("A_bigrams",     (0, dim_A),                  dim_A),
        ("B_morgan",      (dim_A, dim_A + dim_B),      dim_B),
        ("C_descriptors", (dim_A + dim_B, dim_total),   dim_C),
        ("BC_invariant",  (dim_A, dim_total),            dim_B + dim_C),
        ("all",           (0, dim_total),                dim_total),
    ]

    # Train a model for each group (if not already trained)
    eval_subset = test_idx[:500] if len(test_idx) > 500 else test_idx
    results = []

    for group_name, (start, end), dim in groups:
        print(f"\n{'='*60}")
        print(f"Group: {group_name} (dims {start}:{end}, d={dim})")
        print(f"{'='*60}")

        tag = f"mlp_{group_name}"
        ckpt_path = CKPT_DIR / f"{tag}_best.pt"

        # Train if checkpoint doesn't exist
        if not ckpt_path.exists():
            print(f"Training model for {group_name}...")
            run_training(feature_slice=(start, end), feature_group_name=group_name, tag=tag)

        # Load model
        model = MolMLP(dim, mc["hidden_dims"], mc["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Run invariance test
        result = invariance_test_for_group(
            model, features, targets, smiles, eval_subset,
            dim_A, bigram_vocab, feature_mean, feature_std,
            feature_slice=(start, end), group_name=group_name,
            K=30, device=device
        )
        results.append(result)

        print(f"  Canon r_s:  {result['canonical_rs']:.4f}")
        print(f"  Avg r_s:    {result['smiles_averaged_rs']:.4f}")
        print(f"  Mean std:   {result['mean_std']:.4f}")
        print(f"  Consistency:{result['consistency']:.1%}")

    # Save results
    output_path = DATA_DIR / "contamination_decomp.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("CONTAMINATION DECOMPOSITION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Group':<20} {'Canon r_s':>10} {'Avg r_s':>10} {'Std':>8} {'Consist':>8}")
    print(f"{'-'*56}")
    for r in results:
        print(f"{r['group']:<20} {r['canonical_rs']:>10.4f} {r['smiles_averaged_rs']:>10.4f} "
              f"{r['mean_std']:>8.4f} {r['consistency']:>7.1%}")


if __name__ == "__main__":
    main()
