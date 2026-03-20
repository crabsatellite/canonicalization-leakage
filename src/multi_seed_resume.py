"""
Resume multi-seed from where it crashed (7× seed 123).
Runs one experiment at a time with aggressive memory cleanup.
"""

import os
import sys
import json
import time
import gc
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


def train_model(train_feats, train_targets, val_feats, val_targets, device, epochs=60):
    bs = 2048
    train_ds = TensorDataset(
        torch.tensor(train_feats, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(val_feats, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)

    model = CircuitMLP().to(device)
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

    # Clean up training tensors
    del train_ds, val_ds, train_loader, val_loader, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return model, best_epoch


def npn_invariance_test(model, truth_tables, targets, sample_idx,
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

    return {
        "mean_std": float(pred_std.mean()),
        "consistency": float(consistent.mean()),
        "canonical_rs": float(spearmanr(true_targets, all_preds[:, 0])[0]),
        "npn_averaged_rs": float(spearmanr(true_targets, all_preds.mean(axis=1))[0]),
    }


def test_canonical(model, truth_tables, targets, test_idx, measure_mean, measure_std, device):
    feats = compute_features_from_tts(truth_tables[test_idx], measure_mean, measure_std)
    with torch.no_grad():
        preds = model(torch.tensor(feats, dtype=torch.float32).to(device)).cpu().numpy()
    true = targets[test_idx]
    return {
        "spearman_r": float(spearmanr(true, preds)[0]),
        "pearson_r": float(pearsonr(true, preds)[0]),
        "mae": float(np.abs(preds - true).mean()),
        "exact_match": float((np.round(preds) == true).mean()),
    }


def run_single(name, seed, train_feats, train_targets, val_feats, val_targets,
               truth_tables, targets, test_idx, npn_test_idx,
               measure_mean, measure_std, device):
    """Run one experiment with one seed. Returns result dict."""
    print(f"  [{name}] seed={seed}", flush=True)
    set_seed(seed)
    t0 = time.time()

    model, best_epoch = train_model(train_feats, train_targets, val_feats, val_targets, device)
    train_time = time.time() - t0

    test_m = test_canonical(model, truth_tables, targets, test_idx, measure_mean, measure_std, device)
    npn_m = npn_invariance_test(model, truth_tables, targets, npn_test_idx,
                                measure_mean, measure_std, device)

    print(f"    can_rs={test_m['spearman_r']:.4f} npn_avg={npn_m['npn_averaged_rs']:.4f} "
          f"consist={npn_m['consistency']:.4f} ({train_time:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {"test": test_m, "npn_invariance": npn_m, "best_epoch": best_epoch, "train_time_s": train_time}


def main():
    SEEDS = [42, 123, 7]
    data_dir = os.path.join(script_dir, "..", "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = os.path.join(data_dir, "multi_seed_results.json")

    prepared = np.load(os.path.join(data_dir, "prepared.npz"))
    truth_tables = prepared["truth_tables"]
    targets = prepared["targets"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    test_idx = prepared["test_idx"]
    measure_mean = prepared["measure_mean"]
    measure_std = prepared["measure_std"]

    rng_test = np.random.RandomState(42)
    npn_test_idx = rng_test.choice(test_idx, size=500, replace=False)

    val_feats = compute_features_from_tts(truth_tables[val_idx], measure_mean, measure_std)
    val_targets = targets[val_idx]

    # Load existing results from crashed run
    # Hard-code from output since no JSON was saved
    results = {
        "42": {
            "aug_no_augment": {"test": {"spearman_r": 0.7676}, "npn_invariance": {"consistency": 0.0, "npn_averaged_rs": 0.4107, "canonical_rs": 0.7676, "mean_std": 0.0}},
            "aug_1x_augment": {"test": {"spearman_r": 0.7606}, "npn_invariance": {"consistency": 0.3960, "npn_averaged_rs": 0.7201, "canonical_rs": 0.7606, "mean_std": 0.0}},
            "aug_3x_augment": {"test": {"spearman_r": 0.7615}, "npn_invariance": {"consistency": 0.4780, "npn_averaged_rs": 0.7487, "canonical_rs": 0.7615, "mean_std": 0.0}},
            "aug_7x_augment": {"test": {"spearman_r": 0.7751}, "npn_invariance": {"consistency": 0.6180, "npn_averaged_rs": 0.7781, "canonical_rs": 0.7751, "mean_std": 0.0}},
        },
        "123": {
            "aug_no_augment": {"test": {"spearman_r": 0.7664}, "npn_invariance": {"consistency": 0.0, "npn_averaged_rs": 0.2795, "canonical_rs": 0.7664, "mean_std": 0.0}},
            "aug_1x_augment": {"test": {"spearman_r": 0.7578}, "npn_invariance": {"consistency": 0.3960, "npn_averaged_rs": 0.7236, "canonical_rs": 0.7578, "mean_std": 0.0}},
            "aug_3x_augment": {"test": {"spearman_r": 0.7465}, "npn_invariance": {"consistency": 0.6120, "npn_averaged_rs": 0.7390, "canonical_rs": 0.7465, "mean_std": 0.0}},
        },
        "7": {
            "aug_no_augment": {"test": {"spearman_r": 0.7646}, "npn_invariance": {"consistency": 0.0, "npn_averaged_rs": 0.3216, "canonical_rs": 0.7646, "mean_std": 0.0}},
            "aug_1x_augment": {"test": {"spearman_r": 0.7617}, "npn_invariance": {"consistency": 0.4080, "npn_averaged_rs": 0.7285, "canonical_rs": 0.7617, "mean_std": 0.0}},
            "aug_3x_augment": {"test": {"spearman_r": 0.7644}, "npn_invariance": {"consistency": 0.4740, "npn_averaged_rs": 0.7543, "canonical_rs": 0.7644, "mean_std": 0.0}},
        },
    }

    train_tts = truth_tables[train_idx]
    train_targets_arr = targets[train_idx]

    # ======== Remaining: 7× for seeds 123 and 7 ========
    remaining_7x = [123, 7]
    if remaining_7x:
        print(f"\n=== 7x_augment: computing features ===", flush=True)
        t0 = time.time()
        all_tts = [train_tts]
        all_tgts = [train_targets_arr]
        aug_rng = np.random.RandomState(42)
        for _ in range(7):
            all_tts.append(apply_npn_batch(train_tts, aug_rng))
            all_tgts.append(train_targets_arr)
        combined_tts = np.concatenate(all_tts)
        combined_targets = np.concatenate(all_tgts)
        train_feats = compute_features_from_tts(combined_tts, measure_mean, measure_std)
        del all_tts, all_tgts, combined_tts
        gc.collect()
        print(f"  Features: {train_feats.shape[0]:,} in {time.time()-t0:.0f}s", flush=True)

        for seed in remaining_7x:
            r = run_single(f"aug_7x_augment", seed, train_feats, combined_targets,
                          val_feats, val_targets, truth_tables, targets, test_idx,
                          npn_test_idx, measure_mean, measure_std, device)
            results[str(seed)]["aug_7x_augment"] = r

        del train_feats, combined_targets
        gc.collect()
        torch.cuda.empty_cache()

    # ======== Control experiments (all 3 conditions × 3 seeds) ========

    # NPN augmentation (1× copy = 2× volume)
    print(f"\n=== Control: NPN augmentation ===", flush=True)
    aug_rng = np.random.RandomState(42)
    npn_tts = apply_npn_batch(train_tts, aug_rng)
    comb_tts = np.concatenate([train_tts, npn_tts])
    comb_tgts = np.concatenate([train_targets_arr, train_targets_arr])
    feats = compute_features_from_tts(comb_tts, measure_mean, measure_std)
    del comb_tts, npn_tts
    gc.collect()

    for seed in SEEDS:
        r = run_single("ctrl_npn", seed, feats, comb_tgts, val_feats, val_targets,
                       truth_tables, targets, test_idx, npn_test_idx,
                       measure_mean, measure_std, device)
        results[str(seed)]["ctrl_npn_augmentation"] = r

    del feats, comb_tgts
    gc.collect()
    torch.cuda.empty_cache()

    # Duplication
    print(f"\n=== Control: Duplication ===", flush=True)
    dup_tts = np.concatenate([train_tts, train_tts])
    dup_tgts = np.concatenate([train_targets_arr, train_targets_arr])
    feats = compute_features_from_tts(dup_tts, measure_mean, measure_std)
    del dup_tts
    gc.collect()

    for seed in SEEDS:
        r = run_single("ctrl_dup", seed, feats, dup_tgts, val_feats, val_targets,
                       truth_tables, targets, test_idx, npn_test_idx,
                       measure_mean, measure_std, device)
        results[str(seed)]["ctrl_duplication"] = r

    del feats, dup_tgts
    gc.collect()
    torch.cuda.empty_cache()

    # Noise augmentation
    print(f"\n=== Control: Noise augmentation ===", flush=True)
    canonical_feats = compute_features_from_tts(train_tts, measure_mean, measure_std)
    noise_rng = np.random.RandomState(42)
    noise = noise_rng.normal(0, 0.1, size=canonical_feats.shape).astype(np.float32)
    noisy_feats = canonical_feats + noise
    feats = np.concatenate([canonical_feats, noisy_feats])
    tgts = np.concatenate([train_targets_arr, train_targets_arr])
    del canonical_feats, noisy_feats, noise
    gc.collect()

    for seed in SEEDS:
        r = run_single("ctrl_noise", seed, feats, tgts, val_feats, val_targets,
                       truth_tables, targets, test_idx, npn_test_idx,
                       measure_mean, measure_std, device)
        results[str(seed)]["ctrl_noise_augmentation"] = r

    del feats, tgts
    gc.collect()

    # ======== Aggregate ========
    print(f"\n\n{'='*90}")
    print(f"  FINAL MULTI-SEED RESULTS")
    print(f"{'='*90}")

    summary = {}
    all_exp_names = list(results["42"].keys())

    for exp_name in all_exp_names:
        vals = {
            "canonical_rs": [results[str(s)][exp_name]["test"]["spearman_r"] for s in SEEDS],
            "npn_averaged_rs": [results[str(s)][exp_name]["npn_invariance"]["npn_averaged_rs"] for s in SEEDS],
            "consistency": [results[str(s)][exp_name]["npn_invariance"]["consistency"] for s in SEEDS],
        }
        summary[exp_name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for k, v in vals.items()
        }
        m = summary[exp_name]
        print(f"  {exp_name:>25s} | can_rs: {m['canonical_rs']['mean']:.4f}±{m['canonical_rs']['std']:.4f} | "
              f"npn_avg: {m['npn_averaged_rs']['mean']:.4f}±{m['npn_averaged_rs']['std']:.4f} | "
              f"consist: {m['consistency']['mean']:.4f}±{m['consistency']['std']:.4f}")

    output = {"seeds": SEEDS, "per_seed": results, "summary": summary}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Paper numbers
    s = summary
    print(f"\n  PAPER APPENDIX:")
    print(f"  Baseline canonical r_s:    {s['aug_no_augment']['canonical_rs']['mean']:.3f} ± {s['aug_no_augment']['canonical_rs']['std']:.3f}")
    print(f"  Baseline NPN consistency:  {s['aug_no_augment']['consistency']['mean']:.3f} ± {s['aug_no_augment']['consistency']['std']:.3f}")
    print(f"  Baseline NPN-avg r_s:      {s['aug_no_augment']['npn_averaged_rs']['mean']:.3f} ± {s['aug_no_augment']['npn_averaged_rs']['std']:.3f}")
    print(f"  7× aug NPN-avg r_s:        {s['aug_7x_augment']['npn_averaged_rs']['mean']:.3f} ± {s['aug_7x_augment']['npn_averaged_rs']['std']:.3f}")

    aug_order = ["aug_no_augment", "aug_1x_augment", "aug_3x_augment", "aug_7x_augment"]
    means = [s[a]["npn_averaged_rs"]["mean"] for a in aug_order]
    print(f"  Monotonic: {all(means[i] < means[i+1] for i in range(len(means)-1))} {[f'{m:.4f}' for m in means]}")

    ctrl_order = ["ctrl_npn_augmentation", "ctrl_noise_augmentation", "ctrl_duplication"]
    ctrl_means = [s[c]["npn_averaged_rs"]["mean"] for c in ctrl_order]
    print(f"  Control (NPN>noise>dup): {ctrl_means[0] > ctrl_means[1] > ctrl_means[2]} {[f'{m:.4f}' for m in ctrl_means]}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
