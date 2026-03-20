"""
Multi-seed experiment runner.

Runs the key experiments across 3 seeds (42, 123, 7) to report mean ± std.
Data split is fixed (seed=42 in prepared.npz). Only model initialization,
training shuffle, and augmentation RNG vary across seeds.

Experiments:
  1. Baseline MLP (canonical) → test r_s
  2. NPN invariance test → consistency, canonical_rs, npn_averaged_rs
  3. Augmentation sweep (0×, 1×, 3×, 7×) → npn_averaged_rs per level
  4. Matched-volume control (NPN vs noise vs duplication) → npn_averaged_rs
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


# ---- Reused from train_npn_augmented.py ----

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_and_evaluate(train_feats, train_targets, val_feats, val_targets,
                       device, epochs=60, bs=2048):
    """Train MLP return best model state and val spearman."""
    train_ds = TensorDataset(
        torch.tensor(train_feats, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(val_feats, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

    model = CircuitMLP(input_dim=74, hidden_dims=[256, 256, 128]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_r = -1
    best_state = None
    best_epoch = 0

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
        val_r = spearmanr(np.concatenate(trues), np.concatenate(preds))[0]

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def npn_invariance_test(model, truth_tables, targets, sample_idx,
                        measure_mean, measure_std, device, K=30):
    """NPN invariance test on sample_idx."""
    rng = np.random.RandomState(99)  # Fixed for test transforms
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


def test_on_canonical(model, truth_tables, targets, test_idx,
                      measure_mean, measure_std, device):
    feats = compute_features_from_tts(truth_tables[test_idx], measure_mean, measure_std)
    with torch.no_grad():
        X = torch.tensor(feats, dtype=torch.float32).to(device)
        preds = model(X).cpu().numpy()
    true = targets[test_idx]
    return {
        "spearman_r": float(spearmanr(true, preds)[0]),
        "pearson_r": float(pearsonr(true, preds)[0]),
        "mae": float(np.abs(preds - true).mean()),
        "exact_match": float((np.round(preds) == true).mean()),
    }


# ---- Main ----

def main():
    SEEDS = [42, 123, 7]
    data_dir = os.path.join(script_dir, "..", "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    # Fixed NPN test sample
    rng_test = np.random.RandomState(42)
    npn_test_idx = rng_test.choice(test_idx, size=500, replace=False)

    # Pre-compute val features (constant across seeds)
    val_feats = compute_features_from_tts(truth_tables[val_idx], measure_mean, measure_std)

    all_results = {s: {} for s in SEEDS}

    # ======== Experiment 1: Augmentation sweep (includes baseline as 0×) ========
    aug_levels = [0, 1, 3, 7]

    for n_aug in aug_levels:
        name = f"{n_aug}x_augment" if n_aug > 0 else "no_augment"
        print(f"\n{'='*70}")
        print(f"  AUGMENTATION SWEEP: {name}")
        print(f"{'='*70}")

        # Pre-compute augmented training features (augmentation RNG is separate)
        # Use a fixed aug_seed=42 for NPN transforms so all seeds get same augmented data
        # Only model init + shuffle differ across seeds
        print(f"  Computing training features ({name})...", flush=True)
        t0 = time.time()
        train_tts = truth_tables[train_idx]
        train_targets_arr = targets[train_idx]

        all_train_tts = [train_tts]
        all_train_targets = [train_targets_arr]
        aug_rng = np.random.RandomState(42)
        for _ in range(n_aug):
            aug_tts = apply_npn_batch(train_tts, aug_rng)
            all_train_tts.append(aug_tts)
            all_train_targets.append(train_targets_arr)

        combined_tts = np.concatenate(all_train_tts)
        combined_targets = np.concatenate(all_train_targets)
        train_feats = compute_features_from_tts(combined_tts, measure_mean, measure_std)
        feat_time = time.time() - t0
        print(f"  Features: {train_feats.shape[0]:,} samples in {feat_time:.0f}s")

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            set_seed(seed)
            t0 = time.time()

            model, best_epoch = train_and_evaluate(
                train_feats, combined_targets, val_feats, targets[val_idx],
                device, epochs=60)

            train_time = time.time() - t0
            print(f"    Trained in {train_time:.0f}s (best epoch {best_epoch})")

            # Test on canonical
            test_metrics = test_on_canonical(
                model, truth_tables, targets, test_idx,
                measure_mean, measure_std, device)
            print(f"    canonical r_s = {test_metrics['spearman_r']:.4f}")

            # NPN invariance
            npn_res = npn_invariance_test(
                model, truth_tables, targets, npn_test_idx,
                measure_mean, measure_std, device)
            print(f"    NPN: consistency={npn_res['consistency']:.4f}, "
                  f"npn_avg_rs={npn_res['npn_averaged_rs']:.4f}")

            all_results[seed][f"aug_{name}"] = {
                "test": test_metrics,
                "npn_invariance": npn_res,
                "best_epoch": best_epoch,
                "train_time_s": train_time,
            }

        # Free memory
        del train_feats, combined_tts, combined_targets
        torch.cuda.empty_cache()

    # ======== Experiment 2: Matched-volume control (2× = NPN vs noise vs dup) ========
    print(f"\n\n{'='*70}")
    print(f"  MATCHED-VOLUME CONTROL (2× volume)")
    print(f"{'='*70}")

    train_tts = truth_tables[train_idx]
    train_targets_arr = targets[train_idx]
    N_train = len(train_idx)

    conditions = {}

    # --- NPN augmentation (1× NPN copy) ---
    print(f"\n  Computing NPN augmentation features...", flush=True)
    aug_rng = np.random.RandomState(42)
    npn_tts = apply_npn_batch(train_tts, aug_rng)
    npn_combined_tts = np.concatenate([train_tts, npn_tts])
    npn_combined_targets = np.concatenate([train_targets_arr, train_targets_arr])
    npn_feats = compute_features_from_tts(npn_combined_tts, measure_mean, measure_std)
    conditions["npn_augmentation"] = (npn_feats, npn_combined_targets)

    # --- Duplication ---
    print(f"  Computing duplication features...", flush=True)
    dup_tts = np.concatenate([train_tts, train_tts])
    dup_targets = np.concatenate([train_targets_arr, train_targets_arr])
    dup_feats = compute_features_from_tts(dup_tts, measure_mean, measure_std)
    conditions["duplication"] = (dup_feats, dup_targets)

    # --- Noise augmentation ---
    print(f"  Computing noise augmentation features...", flush=True)
    canonical_feats = compute_features_from_tts(train_tts, measure_mean, measure_std)
    noise_rng = np.random.RandomState(42)
    noise = noise_rng.normal(0, 0.1, size=canonical_feats.shape).astype(np.float32)
    noisy_feats = canonical_feats + noise
    noise_combined_feats = np.concatenate([canonical_feats, noisy_feats])
    noise_combined_targets = np.concatenate([train_targets_arr, train_targets_arr])
    conditions["noise_augmentation"] = (noise_combined_feats, noise_combined_targets)

    for cond_name, (cond_feats, cond_targets) in conditions.items():
        print(f"\n  {cond_name}: {cond_feats.shape[0]:,} samples")

        for seed in SEEDS:
            print(f"    --- Seed {seed} ---")
            set_seed(seed)
            t0 = time.time()

            model, best_epoch = train_and_evaluate(
                cond_feats, cond_targets, val_feats, targets[val_idx],
                device, epochs=60)

            train_time = time.time() - t0
            test_metrics = test_on_canonical(
                model, truth_tables, targets, test_idx,
                measure_mean, measure_std, device)
            npn_res = npn_invariance_test(
                model, truth_tables, targets, npn_test_idx,
                measure_mean, measure_std, device)

            print(f"      canonical_rs={test_metrics['spearman_r']:.4f}, "
                  f"npn_avg_rs={npn_res['npn_averaged_rs']:.4f}, "
                  f"consist={npn_res['consistency']:.4f} ({train_time:.0f}s)")

            all_results[seed][f"ctrl_{cond_name}"] = {
                "test": test_metrics,
                "npn_invariance": npn_res,
                "best_epoch": best_epoch,
                "train_time_s": train_time,
            }

    # ======== Aggregate & Report ========
    print(f"\n\n{'='*90}")
    print(f"  MULTI-SEED RESULTS (seeds: {SEEDS})")
    print(f"{'='*90}")

    summary = {}
    all_exp_names = list(all_results[SEEDS[0]].keys())

    for exp_name in all_exp_names:
        vals = {
            "canonical_rs": [all_results[s][exp_name]["test"]["spearman_r"] for s in SEEDS],
            "npn_averaged_rs": [all_results[s][exp_name]["npn_invariance"]["npn_averaged_rs"] for s in SEEDS],
            "consistency": [all_results[s][exp_name]["npn_invariance"]["consistency"] for s in SEEDS],
        }
        summary[exp_name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for k, v in vals.items()
        }

        can_m, can_s = np.mean(vals["canonical_rs"]), np.std(vals["canonical_rs"])
        npn_m, npn_s = np.mean(vals["npn_averaged_rs"]), np.std(vals["npn_averaged_rs"])
        con_m, con_s = np.mean(vals["consistency"]), np.std(vals["consistency"])

        print(f"  {exp_name:>25s} | canonical_rs: {can_m:.4f}±{can_s:.4f} | "
              f"npn_avg_rs: {npn_m:.4f}±{npn_s:.4f} | "
              f"consistency: {con_m:.4f}±{con_s:.4f}")

    # Save
    output = {
        "seeds": SEEDS,
        "per_seed": {str(s): all_results[s] for s in SEEDS},
        "summary": summary,
    }
    out_path = os.path.join(data_dir, "multi_seed_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Key numbers for paper
    print(f"\n{'='*70}")
    print(f"  PAPER APPENDIX NUMBERS")
    print(f"{'='*70}")
    s = summary
    print(f"  Baseline canonical r_s:     "
          f"{s['aug_no_augment']['canonical_rs']['mean']:.3f} ± {s['aug_no_augment']['canonical_rs']['std']:.3f}")
    print(f"  Baseline NPN consistency:   "
          f"{s['aug_no_augment']['consistency']['mean']:.3f} ± {s['aug_no_augment']['consistency']['std']:.3f}")
    print(f"  Baseline NPN-averaged r_s:  "
          f"{s['aug_no_augment']['npn_averaged_rs']['mean']:.3f} ± {s['aug_no_augment']['npn_averaged_rs']['std']:.3f}")
    print(f"  7× aug NPN-averaged r_s:    "
          f"{s['aug_7x_augment']['npn_averaged_rs']['mean']:.3f} ± {s['aug_7x_augment']['npn_averaged_rs']['std']:.3f}")

    # Monotonic check
    aug_order = ["aug_no_augment", "aug_1x_augment", "aug_3x_augment", "aug_7x_augment"]
    means = [s[a]["npn_averaged_rs"]["mean"] for a in aug_order]
    monotonic = all(means[i] < means[i+1] for i in range(len(means)-1))
    print(f"  Monotonic increase:         {monotonic} ({[f'{m:.4f}' for m in means]})")

    # Control ordering
    ctrl_order = ["ctrl_npn_augmentation", "ctrl_noise_augmentation", "ctrl_duplication"]
    ctrl_means = [s[c]["npn_averaged_rs"]["mean"] for c in ctrl_order]
    ctrl_correct = ctrl_means[0] > ctrl_means[1] > ctrl_means[2]
    print(f"  Control ordering (NPN>noise>dup): {ctrl_correct} ({[f'{m:.4f}' for m in ctrl_means]})")


if __name__ == "__main__":
    t_start = time.time()
    main()
    total = time.time() - t_start
    print(f"\nTotal runtime: {total/60:.1f} minutes")
