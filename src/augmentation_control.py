"""
Augmentation Control Experiment: Is the gain from symmetry or from regularization?

Three conditions at matched data volume (2× = canonical + 1 copy):
  1. NPN augmentation:  canonical + 1 NPN-transformed copy (symmetry-consistent)
  2. Data duplication:  canonical repeated 2× (same volume, no new information)
  3. Noise augmentation: canonical + copy with Gaussian noise on features (regularization)

If NPN >> duplication ≈ noise → gain is from symmetry-consistent augmentation
If NPN ≈ duplication ≈ noise → gain is from generic regularization / data volume
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


def apply_npn_batch(tts, rng):
    N = len(tts)
    result = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        perm = list(rng.permutation(5))
        neg_in = rng.randint(0, 32)
        neg_out = bool(rng.randint(0, 2))
        result[i] = apply_npn_transform(int(tts[i]), perm, neg_in, neg_out)
    return result


# ---------- Feature Computation ----------

def compute_features_from_tts(tts, measure_mean, measure_std):
    bits = np.zeros((len(tts), 32), dtype=np.float32)
    for b in range(32):
        bits[:, b] = (tts >> b) & 1

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
    measures_norm = ((measures - measure_mean) / (measure_std + 1e-8)).astype(np.float32)

    return np.concatenate([bits, fourier, measures_norm], axis=1)


# ---------- Model ----------

class CircuitMLP(nn.Module):
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


# ---------- NPN Invariance Test ----------

def npn_invariance_quick(model, truth_tables, targets, sample_idx,
                         measure_mean, measure_std, device, K=30):
    rng = np.random.RandomState(99)
    N = len(sample_idx)
    perms = [list(rng.permutation(5)) for _ in range(K)]
    neg_ins = [rng.randint(0, 32) for _ in range(K)]
    neg_outs = [bool(rng.randint(0, 2)) for _ in range(K)]

    all_preds = np.zeros((N, K + 1))
    tts = truth_tables[sample_idx]

    feats = compute_features_from_tts(tts, measure_mean, measure_std)
    with torch.no_grad():
        X = torch.tensor(feats, dtype=torch.float32).to(device)
        all_preds[:, 0] = model(X).cpu().numpy()

    for k in range(K):
        transformed = np.array([
            apply_npn_transform(int(tt), perms[k], neg_ins[k], neg_outs[k])
            for tt in tts
        ], dtype=np.uint32)
        feats = compute_features_from_tts(transformed, measure_mean, measure_std)
        with torch.no_grad():
            X = torch.tensor(feats, dtype=torch.float32).to(device)
            all_preds[:, k + 1] = model(X).cpu().numpy()

    true_targets = targets[sample_idx]
    pred_std = all_preds.std(axis=1)
    rounded = np.round(all_preds)
    consistent = np.all(rounded == rounded[:, 0:1], axis=1)
    canonical_rs = spearmanr(true_targets, all_preds[:, 0])[0]
    npn_avg_rs = spearmanr(true_targets, all_preds.mean(axis=1))[0]

    return {
        "mean_std": float(pred_std.mean()),
        "consistency": float(consistent.mean()),
        "canonical_rs": float(canonical_rs),
        "npn_averaged_rs": float(npn_avg_rs),
    }


# ---------- Train + Evaluate ----------

def train_and_test(train_feats, train_targets, val_feats, val_targets,
                   epochs=60, bs=2048, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.tensor(train_feats, dtype=torch.float32),
                              torch.tensor(train_targets, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(val_feats, dtype=torch.float32),
                            torch.tensor(val_targets, dtype=torch.float32))
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

    model = CircuitMLP(input_dim=train_feats.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_r = -1
    best_state = None
    for epoch in range(1, epochs + 1):
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

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | val_r_s={r:.4f} | best={best_r:.4f}",
                  flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model, best_r


# ---------- Main ----------

def main():
    data_dir = os.path.join(script_dir, "..", "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    all_features = prepared["features"]
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    rng = np.random.RandomState(42)
    npn_test_idx = rng.choice(test_idx, size=500, replace=False)

    train_feats_canonical = all_features[train_idx]
    train_targets_base = targets[train_idx]
    val_feats = all_features[val_idx]
    val_targets = targets[val_idx]

    # Three conditions, all matched at 2× volume
    conditions = {}

    # --- Condition 1: NPN augmentation (1×) ---
    print(f"\n{'=' * 70}")
    print(f"  Condition 1: NPN augmentation (1× = 2× volume)")
    print(f"{'=' * 70}")
    t0 = time.time()
    aug_tts = apply_npn_batch(truth_tables[train_idx], np.random.RandomState(42))
    aug_feats = compute_features_from_tts(aug_tts, measure_mean, measure_std)
    npn_train_feats = np.concatenate([train_feats_canonical, aug_feats])
    npn_train_targets = np.concatenate([train_targets_base, train_targets_base])
    feat_time = time.time() - t0
    print(f"  Features: {feat_time:.0f}s, shape: {npn_train_feats.shape}", flush=True)

    model, val_r = train_and_test(npn_train_feats, npn_train_targets, val_feats, val_targets)
    npn_inv = npn_invariance_quick(model, truth_tables, targets, npn_test_idx,
                                    measure_mean, measure_std, device)
    print(f"  NPN test: std={npn_inv['mean_std']:.4f}, npn_rs={npn_inv['npn_averaged_rs']:.4f}")
    conditions["npn_augment"] = {"val_rs": float(val_r), **npn_inv}

    # --- Condition 2: Data duplication (canonical repeated 2×) ---
    print(f"\n{'=' * 70}")
    print(f"  Condition 2: Data duplication (canonical × 2)")
    print(f"{'=' * 70}")
    dup_train_feats = np.concatenate([train_feats_canonical, train_feats_canonical])
    dup_train_targets = np.concatenate([train_targets_base, train_targets_base])
    print(f"  Shape: {dup_train_feats.shape} (identical copies)", flush=True)

    model, val_r = train_and_test(dup_train_feats, dup_train_targets, val_feats, val_targets)
    dup_inv = npn_invariance_quick(model, truth_tables, targets, npn_test_idx,
                                    measure_mean, measure_std, device)
    print(f"  NPN test: std={dup_inv['mean_std']:.4f}, npn_rs={dup_inv['npn_averaged_rs']:.4f}")
    conditions["duplication"] = {"val_rs": float(val_r), **dup_inv}

    # --- Condition 3: Gaussian noise augmentation ---
    print(f"\n{'=' * 70}")
    print(f"  Condition 3: Noise augmentation (canonical + Gaussian noise copy)")
    print(f"{'=' * 70}")
    noise_rng = np.random.RandomState(42)
    noise = noise_rng.normal(0, 0.1, size=train_feats_canonical.shape).astype(np.float32)
    noisy_feats = train_feats_canonical + noise
    noise_train_feats = np.concatenate([train_feats_canonical, noisy_feats])
    noise_train_targets = np.concatenate([train_targets_base, train_targets_base])
    print(f"  Shape: {noise_train_feats.shape} (canonical + σ=0.1 noise)", flush=True)

    model, val_r = train_and_test(noise_train_feats, noise_train_targets, val_feats, val_targets)
    noise_inv = npn_invariance_quick(model, truth_tables, targets, npn_test_idx,
                                      measure_mean, measure_std, device)
    print(f"  NPN test: std={noise_inv['mean_std']:.4f}, npn_rs={noise_inv['npn_averaged_rs']:.4f}")
    conditions["noise_augment"] = {"val_rs": float(val_r), **noise_inv}

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print(f"AUGMENTATION CONTROL EXPERIMENT — SUMMARY")
    print(f"{'=' * 80}")
    print(f"All conditions: 2× training volume ({len(train_idx) * 2:,} samples)")
    print(f"")
    print(f"{'Condition':>20s} | {'val_rs':>6s} | {'NPN_std':>7s} | {'consist':>7s} | {'npn_rs':>6s}")
    print(f"{'-' * 65}")

    for name, r in conditions.items():
        print(f"{name:>20s} | {r['val_rs']:>6.4f} | {r['mean_std']:>7.4f} | "
              f"{r['consistency']:>7.4f} | {r['npn_averaged_rs']:>6.4f}")

    print(f"\n  Reference: invariant-measure ceiling r_s = 0.635")

    # Key comparison
    npn_rs = conditions["npn_augment"]["npn_averaged_rs"]
    dup_rs = conditions["duplication"]["npn_averaged_rs"]
    noise_rs = conditions["noise_augment"]["npn_averaged_rs"]

    print(f"\n  KEY RESULT:")
    if npn_rs > dup_rs + 0.05 and npn_rs > noise_rs + 0.05:
        print(f"  NPN augmentation ({npn_rs:.4f}) >> duplication ({dup_rs:.4f}) ≈ noise ({noise_rs:.4f})")
        print(f"  → The gain is from SYMMETRY-CONSISTENT augmentation, not generic regularization.")
        print(f"  → This rules out Reviewer Objection B (\"just data augmentation\").")
    elif npn_rs > dup_rs + 0.02:
        print(f"  NPN augmentation ({npn_rs:.4f}) > duplication ({dup_rs:.4f}), noise ({noise_rs:.4f})")
        print(f"  → Partial evidence for symmetry-specific gain.")
    else:
        print(f"  NPN augmentation ({npn_rs:.4f}) ≈ duplication ({dup_rs:.4f}) ≈ noise ({noise_rs:.4f})")
        print(f"  → Gain may be from generic regularization, not symmetry.")

    print(f"{'=' * 80}")

    out_path = os.path.join(data_dir, "augmentation_control.json")
    with open(out_path, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
