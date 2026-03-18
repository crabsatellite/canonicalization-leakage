"""
Train MLP to predict circuit complexity from truth table features.

Input: 74-dim (32 bits + 32 Fourier + 10 handcrafted measures)
Output: regression (circuit size 0-12)

Also trains classification variant (13 classes) for comparison.
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


class CircuitMLP(nn.Module):
    """MLP for circuit complexity prediction."""

    def __init__(self, input_dim=74, hidden_dims=(256, 256, 128), dropout=0.1,
                 num_classes=None):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h

        self.backbone = nn.Sequential(*layers)

        if num_classes:
            self.head = nn.Linear(prev, num_classes)
        else:
            self.head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

    def get_features(self, x):
        """Extract intermediate features (for interpretability)."""
        return self.backbone(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).squeeze(-1)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(model(X), y.long())
        else:
            loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        n += len(X)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, is_classification=False):
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0
    n = 0

    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        if is_classification:
            pred = out.argmax(dim=-1).float()
            loss = criterion(out, y.long())
        else:
            pred = out.squeeze(-1)
            loss = criterion(pred, y)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
        total_loss += loss.item() * len(X)
        n += len(X)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    mse = ((pred - true) ** 2).mean()
    mae = np.abs(pred - true).mean()
    r_s = spearmanr(true, pred)[0]
    r_p = pearsonr(true, pred)[0]

    # Exact match (rounded prediction)
    exact = (np.round(pred) == true).mean()

    return {
        "loss": total_loss / n,
        "mse": float(mse),
        "mae": float(mae),
        "spearman_r": float(r_s),
        "pearson_r": float(r_p),
        "exact_match": float(exact),
    }


def train_model(mode="regression", config=None):
    """Train MLP in regression or classification mode."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if config is None:
        with open(os.path.join(script_dir, "..", "configs", "default.json")) as f:
            config = json.load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = np.load(os.path.join(script_dir, "..", "data", "prepared.npz"))
    features = data["features"]
    targets = data["targets"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    mc = config["models"]["mlp"]
    bs = mc["batch_size"]

    # DataLoaders
    def make_loader(idx, shuffle):
        X = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(targets[idx], dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    # Model
    is_cls = (mode == "classification")
    model = CircuitMLP(
        input_dim=features.shape[1],
        hidden_dims=mc["hidden_dims"],
        dropout=mc["dropout"],
        num_classes=13 if is_cls else None,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {mode}, params: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mc["epochs"])

    if is_cls:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Training loop
    best_val_r = -1
    best_epoch = 0
    history = []

    t0 = time.time()
    for epoch in range(1, mc["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, is_cls)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["spearman_r"] > best_val_r:
            best_val_r = val_metrics["spearman_r"]
            best_epoch = epoch
            ckpt_path = os.path.join(script_dir, "..", "checkpoints", f"mlp_{mode}_best.pt")
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{mc['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_r_s={val_metrics['spearman_r']:.4f} | "
                  f"val_mae={val_metrics['mae']:.4f} | "
                  f"val_exact={val_metrics['exact_match']:.4f} | "
                  f"{elapsed:.0f}s")

    # Load best and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, is_cls)

    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Training time: {elapsed_total:.1f}s")
    print(f"  Test Spearman r:    {test_metrics['spearman_r']:.4f}")
    print(f"  Test Pearson r:     {test_metrics['pearson_r']:.4f}")
    print(f"  Test MAE:           {test_metrics['mae']:.4f}")
    print(f"  Test Exact match:   {test_metrics['exact_match']:.4f}")
    print(f"  linear baseline r:  {config['linear_baseline_r']:.4f}")
    improvement = test_metrics["pearson_r"] - config["linear_baseline_r"]
    print(f"  Improvement over baseline: {improvement:+.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        "mode": mode,
        "best_epoch": best_epoch,
        "training_time_s": elapsed_total,
        "params": param_count,
        "test_metrics": test_metrics,
        "linear_baseline_r": config["linear_baseline_r"],
        "history": history,
    }
    results_path = os.path.join(script_dir, "..", "data", f"mlp_{mode}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return model, test_metrics


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "regression"
    print(f"\n{'#'*60}")
    print(f"# Training MLP ({mode})")
    print(f"{'#'*60}\n")
    train_model(mode=mode)
