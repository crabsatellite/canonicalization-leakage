"""
NPN-Augmented Training: Force the model to learn NPN-invariant representations.

Each training sample is randomly NPN-transformed on-the-fly:
  - Random permutation of 5 input variables
  - Random negation of each input variable
  - Random output negation

The model sees each function in many NPN-equivalent forms but always with the same
target (circuit complexity), forcing it to learn features invariant to these transforms.

Key question: Does the augmented model outperform the invariant-measures-only model
(r_s ≈ 0.635)? If yes → the MLP extracts NEW invariant features that classical
measures miss. If no → classical measures exhaust the invariant signal.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr

script_dir = os.path.dirname(os.path.abspath(__file__))

from compute_measures import (
    batch_tt_to_bits, walsh_hadamard_transform, shannon_entropy,
    spectral_entropy, batch_lz76, batch_run_length, batch_gzip_ratio,
    batch_algebraic_degree, batch_nonlinearity, batch_autocorrelation_sum,
    batch_sensitivity, batch_influence,
)


# ---------- NPN Transform (vectorized for batches) ----------

def apply_npn_transform(tt, perm, neg_in, neg_out):
    """Apply NPN transform to a single 32-bit truth table."""
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
    """Apply random NPN transforms to a batch of truth tables."""
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
    """Compute full 74-dim features from truth table integers."""
    bits = np.zeros((len(tts), 32), dtype=np.float32)
    for b in range(32):
        bits[:, b] = (tts >> b) & 1

    # Fourier
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
    measures_norm = ((measures - measure_mean) / (measure_std + 1e-8)).astype(np.float32)

    return np.concatenate([bits, fourier, measures_norm], axis=1)


# ---------- Dataset with Online NPN Augmentation ----------

class NPNAugmentedDataset(Dataset):
    """Dataset that applies random NPN transforms on the fly."""

    def __init__(self, truth_tables, targets, measure_mean, measure_std,
                 augment=True, seed=42):
        self.truth_tables = truth_tables.astype(np.uint32)
        self.targets = targets.astype(np.float32)
        self.measure_mean = measure_mean
        self.measure_std = measure_std
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        # Pre-compute canonical features for non-augmented mode
        if not augment:
            self.features = compute_features_from_tts(
                self.truth_tables, measure_mean, measure_std)

    def __len__(self):
        return len(self.truth_tables)

    def __getitem__(self, idx):
        if not self.augment:
            return (torch.tensor(self.features[idx], dtype=torch.float32),
                    torch.tensor(self.targets[idx], dtype=torch.float32))

        # Apply random NPN transform
        tt = int(self.truth_tables[idx])
        perm = list(self.rng.permutation(5))
        neg_in = self.rng.randint(0, 32)
        neg_out = bool(self.rng.randint(0, 2))
        transformed_tt = apply_npn_transform(tt, perm, neg_in, neg_out)

        # Compute features for transformed TT (single sample)
        tts_arr = np.array([transformed_tt], dtype=np.uint32)
        feats = compute_features_from_tts(tts_arr, self.measure_mean, self.measure_std)

        return (torch.tensor(feats[0], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32))


# ---------- Batch-augmented approach (much faster) ----------

class BatchNPNDataset(Dataset):
    """Pre-augments K copies per epoch for speed."""

    def __init__(self, truth_tables, targets, measure_mean, measure_std,
                 n_augment=1, seed=42):
        self.targets_orig = targets.astype(np.float32)
        self.measure_mean = measure_mean
        self.measure_std = measure_std
        self.n_augment = n_augment
        self.rng = np.random.RandomState(seed)
        self.truth_tables_orig = truth_tables.astype(np.uint32)

        # Pre-generate augmented data
        self._generate_epoch()

    def _generate_epoch(self):
        """Generate augmented features for this epoch."""
        N = len(self.truth_tables_orig)

        if self.n_augment == 0:
            # No augmentation — canonical only
            self.features = compute_features_from_tts(
                self.truth_tables_orig, self.measure_mean, self.measure_std)
            self.targets = self.targets_orig
            return

        # Generate augmented copies
        all_tts = []
        all_targets = []

        # Always include canonical
        all_tts.append(self.truth_tables_orig)
        all_targets.append(self.targets_orig)

        # Add augmented copies
        for _ in range(self.n_augment):
            aug_tts = apply_npn_batch(self.truth_tables_orig, self.rng)
            all_tts.append(aug_tts)
            all_targets.append(self.targets_orig)

        combined_tts = np.concatenate(all_tts)
        self.targets = np.concatenate(all_targets)

        # Compute features for all
        self.features = compute_features_from_tts(
            combined_tts, self.measure_mean, self.measure_std)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32))


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
    """Quick NPN invariance test."""
    rng = np.random.RandomState(99)
    N = len(sample_idx)

    perms = [list(rng.permutation(5)) for _ in range(K)]
    neg_ins = [rng.randint(0, 32) for _ in range(K)]
    neg_outs = [bool(rng.randint(0, 2)) for _ in range(K)]

    all_preds = np.zeros((N, K + 1))
    tts = truth_tables[sample_idx]

    # Canonical
    feats = compute_features_from_tts(tts, measure_mean, measure_std)
    with torch.no_grad():
        X = torch.tensor(feats, dtype=torch.float32).to(device)
        all_preds[:, 0] = model(X).cpu().numpy()

    # Transforms
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
        "canonical_mae": float(np.abs(all_preds[:, 0] - true_targets).mean()),
        "npn_averaged_mae": float(np.abs(all_preds.mean(axis=1) - true_targets).mean()),
    }


# ---------- Main ----------

def main():
    data_dir = os.path.join(script_dir, "..", "data")
    ckpt_dir = os.path.join(script_dir, "..", "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    with open(os.path.join(script_dir, "..", "configs", "default.json")) as f:
        config = json.load(f)

    # NPN invariance test sample
    rng = np.random.RandomState(42)
    npn_test_idx = rng.choice(test_idx, size=500, replace=False)

    # Augmentation configs to test
    aug_configs = [
        {"name": "no_augment", "n_augment": 0},
        {"name": "1x_augment", "n_augment": 1},   # canonical + 1 NPN copy = 2×
        {"name": "3x_augment", "n_augment": 3},   # canonical + 3 NPN copies = 4×
        {"name": "7x_augment", "n_augment": 7},   # canonical + 7 NPN copies = 8×
    ]

    results = {}

    for aug_cfg in aug_configs:
        name = aug_cfg["name"]
        n_aug = aug_cfg["n_augment"]
        effective_size = len(train_idx) * (1 + n_aug)

        print(f"\n{'=' * 70}")
        print(f"  {name}: {n_aug} NPN copies + canonical = {effective_size:,} training samples")
        print(f"{'=' * 70}")

        # Feature computation (the slow part)
        print(f"  Computing features...", flush=True)
        t0 = time.time()

        train_tts = truth_tables[train_idx]
        train_targets = targets[train_idx]
        val_tts = truth_tables[val_idx]
        val_targets = targets[val_idx]

        # Build training set
        all_train_tts = [train_tts]
        all_train_targets = [train_targets]
        aug_rng = np.random.RandomState(42)
        for _ in range(n_aug):
            aug_tts = apply_npn_batch(train_tts, aug_rng)
            all_train_tts.append(aug_tts)
            all_train_targets.append(train_targets)

        combined_tts = np.concatenate(all_train_tts)
        combined_targets = np.concatenate(all_train_targets)

        train_feats = compute_features_from_tts(combined_tts, measure_mean, measure_std)
        val_feats = compute_features_from_tts(val_tts, measure_mean, measure_std)

        feat_time = time.time() - t0
        print(f"  Features computed in {feat_time:.0f}s", flush=True)

        # DataLoaders
        bs = 2048
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(train_feats, dtype=torch.float32),
            torch.tensor(combined_targets, dtype=torch.float32))
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(val_feats, dtype=torch.float32),
            torch.tensor(val_targets, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

        # Model
        model = CircuitMLP(input_dim=74, hidden_dims=[256, 256, 128]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        epochs = 60
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        # Train
        print(f"  Training ({epochs} epochs)...", flush=True)
        t0 = time.time()
        best_val_r = -1
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(X), y).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    preds.append(model(X.to(device)).cpu().numpy())
                    trues.append(y.numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            val_r = spearmanr(t, p)[0]

            if val_r > best_val_r:
                best_val_r = val_r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch:3d}/{epochs} | val_r_s={val_r:.4f} | "
                      f"best={best_val_r:.4f}@{best_epoch} | {elapsed:.0f}s", flush=True)

        train_time = time.time() - t0

        # Load best and test
        model.load_state_dict(best_state)
        model.eval()

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"mlp_{name}_best.pt")
        torch.save(best_state, ckpt_path)

        # Test on canonical
        test_feats = compute_features_from_tts(
            truth_tables[test_idx], measure_mean, measure_std)
        with torch.no_grad():
            X = torch.tensor(test_feats, dtype=torch.float32).to(device)
            test_preds = model(X).cpu().numpy()
        test_targets = targets[test_idx]

        test_rs = spearmanr(test_targets, test_preds)[0]
        test_rp = pearsonr(test_targets, test_preds)[0]
        test_mae = np.abs(test_preds - test_targets).mean()
        test_exact = (np.round(test_preds) == test_targets).mean()

        print(f"\n  Test (canonical): r_s={test_rs:.4f}, r_p={test_rp:.4f}, "
              f"MAE={test_mae:.4f}, exact={test_exact:.4f}")

        # NPN invariance test
        print(f"  Running NPN invariance test...", flush=True)
        npn_results = npn_invariance_quick(
            model, truth_tables, targets, npn_test_idx,
            measure_mean, measure_std, device)

        print(f"  NPN: std={npn_results['mean_std']:.4f}, "
              f"consistency={npn_results['consistency']:.4f}, "
              f"can_rs={npn_results['canonical_rs']:.4f}, "
              f"npn_rs={npn_results['npn_averaged_rs']:.4f}")

        results[name] = {
            "n_augment": n_aug,
            "effective_train_size": effective_size,
            "best_epoch": best_epoch,
            "train_time_s": train_time,
            "feat_time_s": feat_time,
            "test": {
                "spearman_r": float(test_rs),
                "pearson_r": float(test_rp),
                "mae": float(test_mae),
                "exact_match": float(test_exact),
            },
            "npn_invariance": npn_results,
        }

    # Summary
    print(f"\n\n{'=' * 90}")
    print(f"NPN AUGMENTATION COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'Config':>15s} | {'Train':>8s} | {'can_rs':>6s} | {'NPN_std':>7s} | "
          f"{'consist':>7s} | {'npn_rs':>6s} | {'MAE':>5s} | {'exact':>5s}")
    print(f"{'-' * 90}")

    for name, r in results.items():
        print(f"{name:>15s} | {r['effective_train_size']:>8,} | "
              f"{r['test']['spearman_r']:>6.4f} | {r['npn_invariance']['mean_std']:>7.4f} | "
              f"{r['npn_invariance']['consistency']:>7.4f} | "
              f"{r['npn_invariance']['npn_averaged_rs']:>6.4f} | "
              f"{r['test']['mae']:>5.3f} | {r['test']['exact_match']:>5.3f}")

    # Reference baselines
    print(f"\n  Reference baselines:")
    print(f"    linear baseline:       r_s = 0.563")
    print(f"    C_invariant (7 measures): r_s = 0.635 (NPN-clean ceiling from classical measures)")
    print(f"    MLP no augment:        r_s = {results['no_augment']['test']['spearman_r']:.4f} "
          f"(includes canonical bias)")

    # Key comparison
    best_aug = max([r for n, r in results.items() if r['n_augment'] > 0],
                   key=lambda x: x['npn_invariance']['npn_averaged_rs'])
    best_aug_name = [n for n, r in results.items()
                     if r['npn_invariance']['npn_averaged_rs'] ==
                     best_aug['npn_invariance']['npn_averaged_rs']][0]

    inv_ceiling = 0.635
    best_npn_rs = best_aug['npn_invariance']['npn_averaged_rs']

    print(f"\n  KEY RESULT:")
    print(f"    Best augmented NPN-averaged r_s: {best_npn_rs:.4f} ({best_aug_name})")
    print(f"    Classical invariant ceiling:     {inv_ceiling:.4f}")
    gap = best_npn_rs - inv_ceiling
    if gap > 0.02:
        print(f"    → Augmented model EXCEEDS classical measures by {gap:+.4f}")
        print(f"    → The MLP learns NEW NPN-invariant features beyond classical measures!")
    elif gap > -0.02:
        print(f"    → Augmented model ≈ classical measures (Δ={gap:+.4f})")
        print(f"    → Classical measures capture most of the invariant signal")
    else:
        print(f"    → Augmented model BELOW classical measures (Δ={gap:+.4f})")
        print(f"    → Training with augmentation hurts; need better architecture")

    print(f"{'=' * 90}")

    # Save
    out_path = os.path.join(data_dir, "npn_augmentation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
