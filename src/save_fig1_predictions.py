"""
Save per-sample predictions for Figure 1.

Runs inference on 500 test functions (matching paper Section 4.2):
  - Canonical predictions
  - K=30 NPN-transformed predictions per function
  - NPN-averaged predictions

Saves to data/fig1_predictions.npz for use by generate_figures.py.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))

from compute_measures import (
    batch_tt_to_bits, walsh_hadamard_transform, shannon_entropy,
    spectral_entropy, batch_lz76, batch_run_length, batch_gzip_ratio,
    batch_algebraic_degree, batch_nonlinearity, batch_autocorrelation_sum,
    batch_sensitivity, batch_influence,
)


class CircuitMLP(nn.Module):
    """Must match train_npn_augmented.py architecture (self.net, not backbone+head)."""
    def __init__(self, input_dim=74, hidden_dims=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def apply_npn_transform(tt, perm, neg_in, neg_out):
    new_tt = 0
    for new_idx in range(32):
        old_idx = 0
        for j in range(5):
            bit = (new_idx >> perm[j]) & 1
            bit ^= (neg_in >> j) & 1
            old_idx |= (bit << j)
        out_bit = (tt >> old_idx) & 1
        if neg_out:
            out_bit ^= 1
        new_tt |= (out_bit << new_idx)
    return new_tt


def compute_features_batch(tts, measure_mean, measure_std):
    bits = batch_tt_to_bits(tts).astype(np.float32)
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
    m_norm = (m - measure_mean) / (measure_std + 1e-8)

    features = np.concatenate([bits, fourier.astype(np.float32), m_norm.astype(np.float32)], axis=1)
    return features


def main():
    data_dir = os.path.join(script_dir, "..", "data")
    ckpt_dir = os.path.join(script_dir, "..", "checkpoints")

    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    with open(os.path.join(script_dir, "..", "configs", "default.json")) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the no_augment model (same one that produced npn_augmentation_results.json)
    model = CircuitMLP(
        input_dim=74,
        hidden_dims=[256, 256, 128],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, "mlp_no_augment_best.pt"),
        map_location=device, weights_only=True))
    model.eval()

    # Match paper exactly: N=500, K=30
    # train_npn_augmented.py uses seed=42 for subset, seed=99 for transforms
    rng_subset = np.random.RandomState(42)
    N_test = 500
    K = 30

    sample_idx = rng_subset.choice(test_idx, size=N_test, replace=False)

    print(f"Running inference: {N_test} functions x {K} NPN transforms...")

    # Generate random transforms with seed=99 (matching npn_invariance_quick)
    rng_transform = np.random.RandomState(99)
    perms = [list(rng_transform.permutation(5)) for _ in range(K)]
    neg_ins = [rng_transform.randint(0, 32) for _ in range(K)]
    neg_outs = [bool(rng_transform.randint(0, 2)) for _ in range(K)]

    canonical_preds = np.zeros(N_test)
    transform_preds = np.zeros((N_test, K))
    true_targets = targets[sample_idx]

    for batch_start in range(0, N_test, 50):
        batch_end = min(batch_start + 50, N_test)
        batch_idx = sample_idx[batch_start:batch_end]

        # Canonical
        canonical_tts = truth_tables[batch_idx]
        canonical_feats = compute_features_batch(canonical_tts, measure_mean, measure_std)
        with torch.no_grad():
            X = torch.tensor(canonical_feats, dtype=torch.float32).to(device)
            pred = model(X).squeeze(-1).cpu().numpy()
        canonical_preds[batch_start:batch_end] = pred

        # NPN transforms
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
            transform_preds[batch_start:batch_end, k] = pred

        print(f"  {batch_end}/{N_test}", flush=True)

    # Compute summary stats
    # Match train_npn_augmented.py: average includes canonical (K+1 predictions)
    all_preds = np.concatenate([canonical_preds.reshape(-1, 1), transform_preds], axis=1)
    npn_averaged_preds = all_preds.mean(axis=1)
    rs_canonical = spearmanr(true_targets, canonical_preds)[0]
    rs_npn_avg = spearmanr(true_targets, npn_averaged_preds)[0]
    mae_canonical = np.abs(canonical_preds - true_targets).mean()
    mae_npn_avg = np.abs(npn_averaged_preds - true_targets).mean()

    print(f"\nResults:")
    print(f"  Canonical:    r_s={rs_canonical:.3f}, MAE={mae_canonical:.3f}")
    print(f"  NPN-averaged: r_s={rs_npn_avg:.3f}, MAE={mae_npn_avg:.3f}")

    out_path = os.path.join(data_dir, "fig1_predictions.npz")
    np.savez(out_path,
             true_targets=true_targets,
             canonical_preds=canonical_preds,
             transform_preds=transform_preds,
             npn_averaged_preds=npn_averaged_preds,
             rs_canonical=rs_canonical,
             rs_npn_avg=rs_npn_avg,
             mae_canonical=mae_canonical,
             mae_npn_avg=mae_npn_avg)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
