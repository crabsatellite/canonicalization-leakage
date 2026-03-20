"""
NPN Contamination Source Decomposition

Identifies which feature components carry canonical bias vs. genuine complexity signal.

Feature groups tested:
  A: bits only (32-dim) — maximally non-invariant (NPN rearranges/flips bits)
  B: Fourier only (32-dim) — partially non-invariant (coefficients permute/sign-change)
  C_all: all 10 measures — mixed invariance
  C_inv: 7 NPN-invariant measures (entropy, spectral_entropy, alg_degree, nonlinearity,
         autocorrelation, sensitivity, influence)
  C_noninv: 3 NPN-non-invariant measures (lz76, run_length, gzip_ratio)

For each group: train model → run NPN invariance test → report.
This identifies whether canonical bias lives in bits, Fourier, or specific measures.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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


def random_npn_transforms(K, rng):
    perms, neg_ins, neg_outs = [], [], []
    for _ in range(K):
        perms.append(list(rng.permutation(5)))
        neg_ins.append(rng.randint(0, 32))
        neg_outs.append(bool(rng.randint(0, 2)))
    return perms, neg_ins, neg_outs


# ---------- Feature Computation ----------

def compute_all_raw_features(tts, return_parts=True):
    """Compute bits, Fourier, and 10 raw measures for an array of truth tables."""
    bits = np.zeros((len(tts), 32), dtype=np.float32)
    for b in range(32):
        bits[:, b] = (tts >> b) & 1

    # Fourier (same as data_prep.py)
    f = 1.0 - 2.0 * bits.copy()
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
    bits_i8 = bits.astype(np.int8)
    wht = walsh_hadamard_transform(bits_i8)

    measures = np.zeros((len(tts), 10), dtype=np.float64)
    measures[:, 0] = shannon_entropy(bits)
    measures[:, 1] = spectral_entropy(wht)
    measures[:, 2] = batch_lz76(bits_i8)
    measures[:, 3] = batch_run_length(bits_i8)
    measures[:, 4] = batch_gzip_ratio(bits_i8)
    measures[:, 5] = batch_algebraic_degree(bits_i8)
    measures[:, 6] = batch_nonlinearity(wht)
    measures[:, 7] = batch_autocorrelation_sum(bits_i8)
    measures[:, 8] = batch_sensitivity(bits_i8)
    measures[:, 9] = batch_influence(bits_i8)

    return bits, fourier, measures


# Measure indices
INVARIANT_MEASURE_IDX = [0, 1, 5, 6, 7, 8, 9]  # entropy, spectral, degree, nonlin, autocorr, sens, influence
NONINVARIANT_MEASURE_IDX = [2, 3, 4]              # lz76, run_length, gzip
MEASURE_NAMES = ["shannon_entropy", "spectral_entropy", "lz76_complexity",
                 "run_length", "gzip_ratio", "algebraic_degree", "nonlinearity",
                 "autocorrelation_sum", "sensitivity", "total_influence"]


# ---------- Model ----------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(features, targets, train_idx, val_idx, epochs=60, bs=2048, lr=0.001):
    """Train and return best model + test predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(idx, shuffle):
        X = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(targets[idx], dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle, pin_memory=True)

    tl, vl = loader(train_idx, True), loader(val_idx, False)

    model = SimpleMLP(features.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_r = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        for X, y in tl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(X), y).backward()
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in vl:
                preds.append(model(X.to(device)).cpu().numpy())
                trues.append(y.numpy())
        r = spearmanr(np.concatenate(trues), np.concatenate(preds))[0]
        if r > best_r:
            best_r = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_r


def npn_invariance_test(model, feature_extractor, truth_tables, targets,
                        sample_idx, perms, neg_ins, neg_outs, device):
    """Run NPN invariance test on a model with a given feature extractor."""
    N = len(sample_idx)
    K = len(perms)

    all_preds = np.zeros((N, K + 1))

    for batch_start in range(0, N, 100):
        batch_end = min(batch_start + 100, N)
        batch_idx = sample_idx[batch_start:batch_end]
        bs = len(batch_idx)

        # Canonical
        canonical_tts = truth_tables[batch_idx]
        feats = feature_extractor(canonical_tts)
        with torch.no_grad():
            X = torch.tensor(feats, dtype=torch.float32).to(device)
            all_preds[batch_start:batch_end, 0] = model(X).cpu().numpy()

        # Transforms
        for k in range(K):
            transformed_tts = np.array([
                apply_npn_transform(int(tt), perms[k], neg_ins[k], neg_outs[k])
                for tt in canonical_tts
            ], dtype=np.uint32)
            feats = feature_extractor(transformed_tts)
            with torch.no_grad():
                X = torch.tensor(feats, dtype=torch.float32).to(device)
                all_preds[batch_start:batch_end, k + 1] = model(X).cpu().numpy()

    true_targets = targets[sample_idx]

    pred_std = all_preds.std(axis=1)
    pred_range = all_preds.max(axis=1) - all_preds.min(axis=1)
    rounded = np.round(all_preds)
    consistent = np.all(rounded == rounded[:, 0:1], axis=1)

    canonical_mae = np.abs(all_preds[:, 0] - true_targets).mean()
    canonical_exact = (np.round(all_preds[:, 0]) == true_targets).mean()
    canonical_r = pearsonr(true_targets, all_preds[:, 0])[0]
    canonical_rs = spearmanr(true_targets, all_preds[:, 0])[0]

    npn_mean_preds = all_preds.mean(axis=1)
    npn_mae = np.abs(npn_mean_preds - true_targets).mean()
    npn_exact = (np.round(npn_mean_preds) == true_targets).mean()
    npn_r = pearsonr(true_targets, npn_mean_preds)[0]
    npn_rs = spearmanr(true_targets, npn_mean_preds)[0]

    return {
        "mean_std": float(pred_std.mean()),
        "median_std": float(np.median(pred_std)),
        "max_std": float(pred_std.max()),
        "mean_range": float(pred_range.mean()),
        "consistency_rate": float(consistent.mean()),
        "canonical_mae": float(canonical_mae),
        "canonical_exact": float(canonical_exact),
        "canonical_r": float(canonical_r),
        "canonical_rs": float(canonical_rs),
        "npn_averaged_mae": float(npn_mae),
        "npn_averaged_exact": float(npn_exact),
        "npn_averaged_r": float(npn_r),
        "npn_averaged_rs": float(npn_rs),
    }


def main():
    data_dir = os.path.join(script_dir, "..", "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    all_features = prepared["features"]
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    # NPN test setup
    rng = np.random.RandomState(42)
    N_test = 500
    K = 30
    sample_idx = rng.choice(test_idx, size=N_test, replace=False)
    perms, neg_ins, neg_outs = random_npn_transforms(K, rng)

    # Feature group definitions
    # all_features layout: [bits(0:32) | fourier(32:64) | measures_standardized(64:74)]
    feature_configs = {
        "A_bits": {
            "cols": list(range(0, 32)),
            "dim": 32,
            "expected_invariance": "NONE (bits rearrange under NPN)",
        },
        "B_fourier": {
            "cols": list(range(32, 64)),
            "dim": 32,
            "expected_invariance": "PARTIAL (coefficients permute/sign-change)",
        },
        "C_all_measures": {
            "cols": list(range(64, 74)),
            "dim": 10,
            "expected_invariance": "MIXED (7 invariant + 3 non-invariant)",
        },
        "C_invariant_measures": {
            "cols": [64 + i for i in INVARIANT_MEASURE_IDX],  # 7 measures
            "dim": 7,
            "expected_invariance": "HIGH (all 7 are NPN-invariant by definition)",
        },
        "C_noninvariant_measures": {
            "cols": [64 + i for i in NONINVARIANT_MEASURE_IDX],  # 3 measures
            "dim": 3,
            "expected_invariance": "NONE (lz76, run_length, gzip depend on bit ordering)",
        },
    }

    results = {}

    for name, cfg in feature_configs.items():
        cols = cfg["cols"]
        feats = all_features[:, cols]

        print(f"{'=' * 70}")
        print(f"  {name} (dim={cfg['dim']}, expected: {cfg['expected_invariance']})")
        print(f"{'=' * 70}")

        # Train
        t0 = time.time()
        print(f"  Training model...", flush=True)
        model, val_rs = train_model(feats, targets, train_idx, val_idx)
        train_time = time.time() - t0
        print(f"  Val r_s = {val_rs:.4f} ({train_time:.0f}s)")

        # Build feature extractor for NPN test
        # Must recompute features from transformed truth tables
        def make_extractor(col_indices, m_mean, m_std):
            def extractor(tts):
                bits, fourier, measures = compute_all_raw_features(tts)
                measures_norm = (measures - m_mean) / (m_std + 1e-8)
                full = np.concatenate([bits, fourier, measures_norm.astype(np.float32)], axis=1)
                return full[:, col_indices].astype(np.float32)
            return extractor

        extractor = make_extractor(cols, measure_mean, measure_std)

        # NPN test
        print(f"  Running NPN invariance test ({N_test} functions × {K} transforms)...", flush=True)
        t0 = time.time()
        inv_results = npn_invariance_test(
            model, extractor, truth_tables, targets,
            sample_idx, perms, neg_ins, neg_outs, device)
        test_time = time.time() - t0

        inv_results["val_spearman_r"] = float(val_rs)
        inv_results["train_time_s"] = train_time
        inv_results["test_time_s"] = test_time
        inv_results["dim"] = cfg["dim"]
        inv_results["expected_invariance"] = cfg["expected_invariance"]
        results[name] = inv_results

        print(f"  NPN test: mean_std={inv_results['mean_std']:.4f}, "
              f"consistency={inv_results['consistency_rate']:.4f}, "
              f"canonical_rs={inv_results['canonical_rs']:.4f}, "
              f"npn_avg_rs={inv_results['npn_averaged_rs']:.4f} "
              f"({test_time:.0f}s)")
        print()

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"CONTAMINATION SOURCE DECOMPOSITION — SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Group':>25s} | {'dim':>3} | {'val_rs':>6} | {'NPN_std':>7} | "
          f"{'consist':>7} | {'can_rs':>6} | {'npn_rs':>6} | {'Δrs':>6}")
    print(f"{'-' * 90}")

    for name, r in results.items():
        delta_rs = r["npn_averaged_rs"] - r["canonical_rs"]
        print(f"{name:>25s} | {r['dim']:>3} | {r['val_spearman_r']:>6.4f} | "
              f"{r['mean_std']:>7.4f} | {r['consistency_rate']:>7.4f} | "
              f"{r['canonical_rs']:>6.4f} | {r['npn_averaged_rs']:>6.4f} | "
              f"{delta_rs:>+6.4f}")

    # Interpretation
    print(f"\n{'=' * 90}")
    print("INTERPRETATION:")

    bits_std = results["A_bits"]["mean_std"]
    fourier_std = results["B_fourier"]["mean_std"]
    inv_meas_std = results["C_invariant_measures"]["mean_std"]
    noninv_meas_std = results["C_noninvariant_measures"]["mean_std"]

    print(f"\n  Contamination hierarchy (NPN prediction std):")
    ranking = sorted(results.items(), key=lambda x: x[1]["mean_std"])
    for name, r in ranking:
        label = "CLEAN" if r["mean_std"] < 0.1 else \
                "LOW BIAS" if r["mean_std"] < 0.3 else \
                "MEDIUM BIAS" if r["mean_std"] < 0.5 else \
                "HIGH BIAS" if r["mean_std"] < 0.8 else "SEVERE BIAS"
        print(f"    {name:>25s}: std={r['mean_std']:.4f}  [{label}]")

    if inv_meas_std < 0.1:
        print(f"\n  KEY FINDING: NPN-invariant measures are CLEAN.")
        print(f"    → The genuine complexity signal lives in classical invariants.")
        print(f"    → Canonical bias is concentrated in bit/Fourier representations.")
    elif inv_meas_std < bits_std * 0.5:
        print(f"\n  KEY FINDING: NPN-invariant measures carry LESS bias than bits/Fourier.")
        print(f"    → Partial genuine signal exists in invariant measures.")
    else:
        print(f"\n  KEY FINDING: Even invariant measures show bias.")
        print(f"    → Standardization or training on canonical forms induces residual bias.")

    print(f"\n  Bits NPN std ({bits_std:.4f}) vs Fourier NPN std ({fourier_std:.4f}):")
    if abs(bits_std - fourier_std) < 0.1:
        print(f"    → Similar bias: Fourier does NOT provide invariance over bits.")
        print(f"    → Both carry canonical representation artifacts.")
    elif fourier_std < bits_std:
        print(f"    → Fourier is more invariant than bits (partial spectral invariance).")
    else:
        print(f"    → Fourier is LESS invariant than bits (unexpected).")

    print(f"{'=' * 90}")

    # Save
    out_path = os.path.join(data_dir, "npn_contamination_decomp.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
