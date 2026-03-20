"""
Ablation study: which feature groups drive the MLP's performance?

Feature groups:
  A: truth_table_bits (32 dim, columns 0-31)
  B: fourier_spectrum  (32 dim, columns 32-63)
  C: handcrafted_measures (10 dim, columns 64-73)

Test combinations: A, B, C, A+B, A+C, B+C, A+B+C (full)
Also test deeper model on full features.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr, pearsonr


class AblationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_eval(features, targets, train_idx, val_idx, test_idx,
                   hidden_dims=(256, 256, 128), epochs=60, bs=2048, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(idx, shuffle):
        X = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(targets[idx], dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle,
                          pin_memory=True)

    tl, vl, tel = loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)

    model = AblationMLP(features.shape[1], hidden_dims).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_r = -1
    for epoch in range(epochs):
        model.train()
        for X, y in tl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in vl:
                preds.append(model(X.to(device)).cpu().numpy())
                trues.append(y.numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        r = spearmanr(t, p)[0]
        if r > best_r:
            best_r = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in tel:
            preds.append(model(X.to(device)).cpu().numpy())
            trues.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)

    return {
        "spearman_r": float(spearmanr(t, p)[0]),
        "pearson_r": float(pearsonr(t, p)[0]),
        "mae": float(np.abs(p - t).mean()),
        "exact_match": float((np.round(p) == t).mean()),
        "params": sum(par.numel() for par in model.parameters()),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(script_dir, "..", "data", "prepared.npz"))
    features = data["features"]
    targets = data["targets"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    # Feature groups
    groups = {
        "A_bits":       (0, 32),
        "B_fourier":    (32, 64),
        "C_measures":   (64, 74),
    }

    configs = {
        "A_bits_only":         ["A_bits"],
        "B_fourier_only":      ["B_fourier"],
        "C_measures_only":     ["C_measures"],
        "AB_bits+fourier":     ["A_bits", "B_fourier"],
        "AC_bits+measures":    ["A_bits", "C_measures"],
        "BC_fourier+measures": ["B_fourier", "C_measures"],
        "ABC_full":            ["A_bits", "B_fourier", "C_measures"],
    }

    # Also test deeper architecture on full features
    arch_configs = {
        "ABC_shallow(128,64)":        {"hidden": [128, 64], "groups": ["A_bits", "B_fourier", "C_measures"]},
        "ABC_standard(256,256,128)":  {"hidden": [256, 256, 128], "groups": ["A_bits", "B_fourier", "C_measures"]},
        "ABC_deep(512,512,256,128)":  {"hidden": [512, 512, 256, 128], "groups": ["A_bits", "B_fourier", "C_measures"]},
        "ABC_wide(512,512,512)":      {"hidden": [512, 512, 512], "groups": ["A_bits", "B_fourier", "C_measures"]},
    }

    results = {}

    # Feature ablation (standard architecture)
    print("=" * 70)
    print("FEATURE ABLATION (hidden=[256,256,128], 60 epochs)")
    print("=" * 70)

    for name, group_list in configs.items():
        cols = []
        for g in group_list:
            s, e = groups[g]
            cols.extend(range(s, e))
        feats = features[:, cols]
        print(f"\n--- {name} (dim={feats.shape[1]}) ---")
        t0 = time.time()
        r = train_and_eval(feats, targets, train_idx, val_idx, test_idx)
        elapsed = time.time() - t0
        r["dim"] = feats.shape[1]
        r["time_s"] = elapsed
        results[name] = r
        print(f"  r_s={r['spearman_r']:.4f}  r_p={r['pearson_r']:.4f}  "
              f"MAE={r['mae']:.4f}  exact={r['exact_match']:.4f}  ({elapsed:.0f}s)")

    # Architecture ablation
    print(f"\n{'=' * 70}")
    print("ARCHITECTURE ABLATION (full features, 60 epochs)")
    print("=" * 70)

    for name, cfg in arch_configs.items():
        cols = []
        for g in cfg["groups"]:
            s, e = groups[g]
            cols.extend(range(s, e))
        feats = features[:, cols]
        print(f"\n--- {name} (dim={feats.shape[1]}) ---")
        t0 = time.time()
        r = train_and_eval(feats, targets, train_idx, val_idx, test_idx,
                           hidden_dims=cfg["hidden"])
        elapsed = time.time() - t0
        r["dim"] = feats.shape[1]
        r["time_s"] = elapsed
        r["architecture"] = cfg["hidden"]
        results[name] = r
        print(f"  r_s={r['spearman_r']:.4f}  r_p={r['pearson_r']:.4f}  "
              f"MAE={r['mae']:.4f}  exact={r['exact_match']:.4f}  "
              f"params={r['params']:,}  ({elapsed:.0f}s)")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Config':>30s} | {'dim':>4s} | {'r_s':>6s} | {'r_p':>6s} | {'MAE':>6s} | {'exact':>6s} | {'params':>8s}")
    print("-" * 85)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["spearman_r"]):
        print(f"{name:>30s} | {r['dim']:>4d} | {r['spearman_r']:>6.4f} | {r['pearson_r']:>6.4f} | "
              f"{r['mae']:>6.4f} | {r['exact_match']:>6.4f} | {r.get('params', 0):>8,}")

    print(f"\nlinear baseline: r_p = 0.5627")

    # Save
    out_path = os.path.join(script_dir, "..", "data", "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
