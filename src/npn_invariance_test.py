"""
NPN Invariance Test — Watershed Experiment

Tests whether the trained MLP's predictions are invariant under NPN transformations
(input permutation, input negation, output negation).

All NPN-equivalent Boolean functions have identical circuit complexity.
If the model's predictions are invariant → it has learned function-level structural
representation (supports Proposition C: latent structural representation).
If not invariant → it's a pattern matcher on canonical form (Proposition A only).

NPN group for 5 variables: |S_5| × 2^5 × 2 = 120 × 32 × 2 = 7,680 transforms.
"""

import os
import sys
import json
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

script_dir = os.path.dirname(os.path.abspath(__file__))

from compute_measures import (
    batch_tt_to_bits, walsh_hadamard_transform, shannon_entropy,
    spectral_entropy, batch_lz76, batch_run_length, batch_gzip_ratio,
    batch_algebraic_degree, batch_nonlinearity, batch_autocorrelation_sum,
    batch_sensitivity, batch_influence,
)


# ---------- NPN Transform ----------

def apply_npn_transform(tt, perm, neg_in, neg_out):
    """
    Apply NPN transform to a 32-bit truth table.

    g(x0,...,x4) = f(x_{σ(0)} ⊕ ε_0, ..., x_{σ(4)} ⊕ ε_4) ⊕ δ

    Args:
        tt: 32-bit truth table integer
        perm: list of 5 ints, permutation σ
        neg_in: 5-bit int, input negation mask ε
        neg_out: bool, output negation δ
    """
    new_tt = 0
    for new_idx in range(32):
        # For each input position in the new function,
        # find the corresponding position in the original
        old_idx = 0
        for j in range(5):
            bit = (new_idx >> perm[j]) & 1  # x_{σ(j)}
            bit ^= (neg_in >> j) & 1         # ⊕ ε_j
            old_idx |= (bit << j)

        out_bit = (tt >> old_idx) & 1
        if neg_out:
            out_bit ^= 1

        new_tt |= (out_bit << new_idx)
    return new_tt


def apply_npn_batch(tt, perms, neg_ins, neg_outs):
    """Apply multiple NPN transforms to a single truth table. Returns array of TTs."""
    results = np.zeros(len(perms), dtype=np.uint32)
    for k in range(len(perms)):
        results[k] = apply_npn_transform(tt, perms[k], neg_ins[k], neg_outs[k])
    return results


def random_npn_transforms(K, rng):
    """Generate K random NPN transforms."""
    perms = []
    neg_ins = []
    neg_outs = []
    for _ in range(K):
        perm = list(rng.permutation(5))
        perms.append(perm)
        neg_ins.append(rng.randint(0, 32))   # 0..31 (5 bits)
        neg_outs.append(bool(rng.randint(0, 2)))
    return perms, neg_ins, neg_outs


# ---------- Feature Computation ----------

def compute_features_batch(tts, measure_mean, measure_std):
    """
    Compute full 74-dim features for an array of truth tables.
    Matches data_prep.py: [bits(32) | fourier(32) | measures(10)]
    """
    bits = batch_tt_to_bits(tts).astype(np.float32)  # (N, 32)

    # Walsh-Hadamard (Fourier)
    wht = walsh_hadamard_transform(bits.astype(np.int8))
    fourier = (1.0 - 2.0 * bits) / 32.0  # re-derive normalized
    # Actually use the same WHT as data_prep.py
    f = 1.0 - 2.0 * bits
    for i in range(5):
        half = 1 << i
        for j in range(0, 32, 2 * half):
            for k in range(half):
                u = f[:, j + k].copy()
                v = f[:, j + k + half].copy()
                f[:, j + k] = u + v
                f[:, j + k + half] = u - v
    fourier = f / 32.0

    # Measures
    bits_int8 = bits.astype(np.int8)
    wht_full = walsh_hadamard_transform(bits_int8)

    m = np.zeros((len(tts), 10), dtype=np.float64)
    m[:, 0] = shannon_entropy(bits)
    m[:, 1] = spectral_entropy(wht_full)
    m[:, 2] = batch_lz76(bits_int8)
    m[:, 3] = batch_run_length(bits_int8)
    m[:, 4] = batch_gzip_ratio(bits_int8)
    m[:, 5] = batch_algebraic_degree(bits_int8)
    m[:, 6] = batch_nonlinearity(wht_full)
    m[:, 7] = batch_autocorrelation_sum(bits_int8)
    m[:, 8] = batch_sensitivity(bits_int8)
    m[:, 9] = batch_influence(bits_int8)

    # Standardize measures using training set statistics
    m_norm = (m - measure_mean) / (measure_std + 1e-8)

    features = np.concatenate([bits, fourier.astype(np.float32), m_norm.astype(np.float32)], axis=1)
    return features


# ---------- Model ----------

class CircuitMLP(nn.Module):
    def __init__(self, input_dim=74, hidden_dims=(256, 256, 128), dropout=0.1,
                 num_classes=None):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout),
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes if num_classes else 1)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------- Main Test ----------

def main():
    data_dir = os.path.join(script_dir, "..", "data")
    ckpt_dir = os.path.join(script_dir, "..", "checkpoints")

    # Load prepared data
    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    # Load config
    with open(os.path.join(script_dir, "..", "configs", "default.json")) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    mc = config["models"]["mlp"]
    model = CircuitMLP(
        input_dim=74,
        hidden_dims=mc["hidden_dims"],
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, "mlp_regression_best.pt"), weights_only=True))
    model.eval()

    # Sample test functions
    rng = np.random.RandomState(42)
    N_test = 1000
    sample_idx = rng.choice(test_idx, size=N_test, replace=False)

    K = 50  # NPN transforms per function

    print(f"NPN Invariance Test")
    print(f"  Model: MLP regression (best checkpoint)")
    print(f"  Test functions: {N_test}")
    print(f"  NPN transforms per function: {K}")
    print(f"  Total forward passes: {N_test * (K + 1):,}")
    print(f"  Device: {device}")
    print()

    # Generate random NPN transforms (same for all functions)
    perms, neg_ins, neg_outs = random_npn_transforms(K, rng)

    # Results storage
    all_canonical_preds = []
    all_transform_preds = []  # (N_test, K)
    all_targets = []

    t0 = time.time()

    for batch_start in range(0, N_test, 50):
        batch_end = min(batch_start + 50, N_test)
        batch_idx = sample_idx[batch_start:batch_end]
        batch_size = len(batch_idx)

        # Canonical predictions
        canonical_tts = truth_tables[batch_idx]
        canonical_feats = compute_features_batch(canonical_tts, measure_mean, measure_std)
        with torch.no_grad():
            X = torch.tensor(canonical_feats, dtype=torch.float32).to(device)
            canonical_pred = model(X).squeeze(-1).cpu().numpy()
        all_canonical_preds.append(canonical_pred)
        all_targets.append(targets[batch_idx])

        # NPN-transformed predictions
        batch_transform_preds = np.zeros((batch_size, K))
        for k in range(K):
            transformed_tts = np.array([
                apply_npn_transform(int(tt), perms[k], neg_ins[k], neg_outs[k])
                for tt in canonical_tts
            ], dtype=np.uint32)
            transformed_feats = compute_features_batch(
                transformed_tts, measure_mean, measure_std)
            with torch.no_grad():
                X = torch.tensor(transformed_feats, dtype=torch.float32).to(device)
                pred = model(X).squeeze(-1).cpu().numpy()
            batch_transform_preds[:, k] = pred
        all_transform_preds.append(batch_transform_preds)

        elapsed = time.time() - t0
        print(f"  Processed {batch_end}/{N_test} functions ({elapsed:.0f}s)", flush=True)

    # Aggregate
    canonical_preds = np.concatenate(all_canonical_preds)
    transform_preds = np.vstack(all_transform_preds)  # (N_test, K)
    true_targets = np.concatenate(all_targets)

    # Include canonical in the full set
    all_preds = np.concatenate([canonical_preds.reshape(-1, 1), transform_preds], axis=1)  # (N_test, K+1)

    elapsed_total = time.time() - t0

    # ---------- Analysis ----------
    print(f"\n{'=' * 70}")
    print(f"NPN INVARIANCE RESULTS")
    print(f"{'=' * 70}")

    # 1. Prediction variance across NPN equivalents
    pred_std = all_preds.std(axis=1)          # per-function std across transforms
    pred_range = all_preds.max(axis=1) - all_preds.min(axis=1)  # max spread
    pred_mean = all_preds.mean(axis=1)        # mean prediction

    print(f"\n1. Prediction spread across NPN equivalents:")
    print(f"   Mean std:   {pred_std.mean():.4f} gates")
    print(f"   Median std: {np.median(pred_std):.4f} gates")
    print(f"   Max std:    {pred_std.max():.4f} gates")
    print(f"   Mean range: {pred_range.mean():.4f} gates")
    print(f"   Max range:  {pred_range.max():.4f} gates")

    # 2. Exact match consistency
    rounded_preds = np.round(all_preds)
    consistent = np.all(rounded_preds == rounded_preds[:, 0:1], axis=1)
    consistency_rate = consistent.mean()
    print(f"\n2. Rounded prediction consistency:")
    print(f"   Functions where ALL {K+1} NPN variants agree: {consistency_rate:.4f} ({consistent.sum()}/{N_test})")

    # Also check: does the canonical prediction agree with majority of transforms?
    mode_pred = np.round(all_preds).astype(int)
    from scipy.stats import mode as scipy_mode
    modes = scipy_mode(mode_pred, axis=1, keepdims=False)
    mode_agreement = (mode_pred == modes.mode.reshape(-1, 1)).mean(axis=1)
    print(f"   Mean majority agreement rate: {mode_agreement.mean():.4f}")

    # 3. Accuracy degradation
    canonical_mae = np.abs(canonical_preds - true_targets).mean()
    canonical_exact = (np.round(canonical_preds) == true_targets).mean()
    transform_mae = np.abs(transform_preds.mean(axis=1) - true_targets).mean()
    transform_exact = (np.round(transform_preds.mean(axis=1)) == true_targets).mean()

    print(f"\n3. Accuracy comparison:")
    print(f"   Canonical:   MAE={canonical_mae:.4f}, exact={canonical_exact:.4f}")
    print(f"   NPN-averaged: MAE={transform_mae:.4f}, exact={transform_exact:.4f}")
    improvement = transform_mae - canonical_mae
    print(f"   MAE change:   {improvement:+.4f} ({'worse' if improvement > 0 else 'BETTER'})")

    # 4. Variance by circuit complexity
    print(f"\n4. Prediction spread by circuit complexity:")
    print(f"   {'Size':>4} | {'Count':>5} | {'Mean std':>8} | {'Mean range':>10} | {'Consistent':>10}")
    print(f"   {'-'*50}")
    for size in sorted(np.unique(true_targets)):
        mask = true_targets == size
        if mask.sum() < 5:
            continue
        print(f"   {int(size):>4} | {mask.sum():>5} | {pred_std[mask].mean():>8.4f} | "
              f"{pred_range[mask].mean():>10.4f} | {consistent[mask].mean():>10.4f}")

    # 5. Which NPN component causes most variance?
    print(f"\n5. Variance decomposition by NPN component:")
    # Test permutation only (no negation)
    perm_only_preds = []
    neg_only_preds = []
    for k in range(K):
        is_perm_only = (neg_ins[k] == 0 and not neg_outs[k])
        is_neg_only = (perms[k] == list(range(5)))
        if is_perm_only:
            perm_only_preds.append(k)
        if is_neg_only:
            neg_only_preds.append(k)

    if len(perm_only_preds) > 1:
        p_std = transform_preds[:, perm_only_preds].std(axis=1)
        print(f"   Permutation only ({len(perm_only_preds)} samples): mean std = {p_std.mean():.4f}")
    else:
        print(f"   Permutation only: insufficient samples (got {len(perm_only_preds)})")

    if len(neg_only_preds) > 1:
        n_std = transform_preds[:, neg_only_preds].std(axis=1)
        print(f"   Negation only ({len(neg_only_preds)} samples): mean std = {n_std.mean():.4f}")
    else:
        print(f"   Negation only: insufficient samples (got {len(neg_only_preds)})")

    print(f"   Full NPN: mean std = {pred_std.mean():.4f}")

    # 6. Correlation preservation
    r_canonical = pearsonr(true_targets, canonical_preds)[0]
    r_mean_npn = pearsonr(true_targets, all_preds.mean(axis=1))[0]
    rs_canonical = spearmanr(true_targets, canonical_preds)[0]
    rs_mean_npn = spearmanr(true_targets, all_preds.mean(axis=1))[0]

    print(f"\n6. Correlation with true complexity:")
    print(f"   Canonical:    r_p={r_canonical:.4f}, r_s={rs_canonical:.4f}")
    print(f"   NPN-averaged: r_p={r_mean_npn:.4f}, r_s={rs_mean_npn:.4f}")

    # Overall verdict
    print(f"\n{'=' * 70}")
    if pred_std.mean() < 0.1:
        verdict = "STRONG INVARIANCE"
        interpretation = ("The model has learned NPN-invariant representations. "
                         "Circuit complexity prediction is robust to function representation. "
                         "Supports Proposition C: latent structural representation.")
    elif pred_std.mean() < 0.3:
        verdict = "PARTIAL INVARIANCE"
        interpretation = ("The model shows partial NPN invariance. Some representation-"
                         "dependent features leak through, but the core prediction is stable. "
                         "Supports Proposition B with caveats.")
    elif pred_std.mean() < 0.5:
        verdict = "WEAK INVARIANCE"
        interpretation = ("The model shows weak NPN invariance. Predictions vary significantly "
                         "across equivalent representations. The model relies partly on "
                         "representation-specific patterns.")
    else:
        verdict = "NOT INVARIANT"
        interpretation = ("The model is NOT NPN-invariant. Predictions depend strongly on "
                         "the specific truth table representation. The model is a pattern "
                         "matcher, not a complexity estimator. Proposition A only.")

    print(f"VERDICT: {verdict}")
    print(f"  Mean prediction std across NPN equivalents: {pred_std.mean():.4f} gates")
    print(f"  Rounded consistency rate: {consistency_rate:.4f}")
    print(f"  {interpretation}")
    print(f"  Total time: {elapsed_total:.0f}s")
    print(f"{'=' * 70}")

    # Save results
    results = {
        "n_test": N_test,
        "k_transforms": K,
        "total_time_s": elapsed_total,
        "prediction_spread": {
            "mean_std": float(pred_std.mean()),
            "median_std": float(np.median(pred_std)),
            "max_std": float(pred_std.max()),
            "mean_range": float(pred_range.mean()),
            "max_range": float(pred_range.max()),
        },
        "consistency": {
            "exact_rounded_consistency": float(consistency_rate),
            "mean_majority_agreement": float(mode_agreement.mean()),
        },
        "accuracy": {
            "canonical_mae": float(canonical_mae),
            "canonical_exact": float(canonical_exact),
            "npn_averaged_mae": float(transform_mae),
            "npn_averaged_exact": float(transform_exact),
        },
        "correlation": {
            "canonical_pearson": float(r_canonical),
            "canonical_spearman": float(rs_canonical),
            "npn_averaged_pearson": float(r_mean_npn),
            "npn_averaged_spearman": float(rs_mean_npn),
        },
        "verdict": verdict,
    }

    out_path = os.path.join(data_dir, "npn_invariance_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
