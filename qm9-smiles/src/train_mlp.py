"""
MLP training for QM9 molecular property prediction.
Parallels the NPN circuit experiment's train_mlp.py.

Feature groups:
  A: SMILES bigrams (non-invariant) — dims [0, dim_A)
  B: Morgan fingerprints (invariant) — dims [dim_A, dim_A+dim_B)
  C: Molecular descriptors (invariant) — dims [dim_A+dim_B, dim_A+dim_B+dim_C)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"
CONFIG_PATH = PROJECT_DIR / "configs" / "default.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


class MolMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
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
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    n = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = np.mean(np.abs(preds - trues))
    rs, _ = spearmanr(preds, trues)
    rp, _ = pearsonr(preds, trues)
    return {
        "loss": total_loss / n,
        "mae": mae,
        "spearman_r": rs,
        "pearson_r": rp,
    }


def run_training(feature_slice=None, feature_group_name="all", seed=42, tag=None):
    """
    Train MLP on specified feature subset.

    Args:
        feature_slice: tuple (start, end) or None for all features
        feature_group_name: name for logging/saving
        seed: random seed
        tag: optional tag for checkpoint name
    """
    config = load_config()
    mc = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training MLP on feature group: {feature_group_name}")
    print(f"Device: {device}, Seed: {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data = np.load(DATA_DIR / "prepared.npz", allow_pickle=True)
    features = data["features"]
    targets = data["targets"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # Select feature subset
    if feature_slice is not None:
        features = features[:, feature_slice[0]:feature_slice[1]]
    input_dim = features.shape[1]
    print(f"Input dim: {input_dim}")

    # Create dataloaders
    X_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32)
    X_test = torch.tensor(features[test_idx], dtype=torch.float32)
    y_test = torch.tensor(targets[test_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=mc["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=mc["batch_size"])
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=mc["batch_size"])

    # Model
    model = MolMLP(input_dim, mc["hidden_dims"], mc["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mc["lr"],
                                  weight_decay=mc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, mc["epochs"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Training loop
    best_val_rs = -1
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = tag or f"mlp_{feature_group_name}"
    ckpt_path = CKPT_DIR / f"{ckpt_name}_best.pt"

    for epoch in range(mc["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics["spearman_r"] > best_val_rs:
            best_val_rs = val_metrics["spearman_r"]
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                  f"| val_rs={val_metrics['spearman_r']:.4f} "
                  f"| val_mae={val_metrics['mae']:.4f} "
                  f"| best_rs={best_val_rs:.4f}")

    # Load best and evaluate on test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest results ({feature_group_name}):")
    print(f"  Spearman r_s = {test_metrics['spearman_r']:.4f}")
    print(f"  Pearson r    = {test_metrics['pearson_r']:.4f}")
    print(f"  MAE          = {test_metrics['mae']:.4f}")

    return test_metrics, ckpt_path


def main():
    config = load_config()
    data = np.load(DATA_DIR / "prepared.npz", allow_pickle=True)
    dim_A = int(data["dim_A"])
    dim_B = int(data["dim_B"])
    dim_C = int(data["dim_C"])
    dim_total = dim_A + dim_B + dim_C

    print(f"Feature dims: A(bigrams)={dim_A}, B(morgan)={dim_B}, C(desc)={dim_C}, total={dim_total}")

    results = {}

    # Full model (all features)
    metrics, _ = run_training(feature_slice=None, feature_group_name="all", tag="mlp_all")
    results["all"] = metrics

    # Group A only (SMILES bigrams — non-invariant)
    metrics, _ = run_training(feature_slice=(0, dim_A), feature_group_name="A_bigrams", tag="mlp_A")
    results["A_bigrams"] = metrics

    # Group B only (Morgan FP — invariant)
    metrics, _ = run_training(feature_slice=(dim_A, dim_A + dim_B), feature_group_name="B_morgan", tag="mlp_B")
    results["B_morgan"] = metrics

    # Group C only (Descriptors — invariant)
    metrics, _ = run_training(feature_slice=(dim_A + dim_B, dim_total), feature_group_name="C_descriptors", tag="mlp_C")
    results["C_descriptors"] = metrics

    # B+C (all invariant features)
    metrics, _ = run_training(feature_slice=(dim_A, dim_total), feature_group_name="BC_invariant", tag="mlp_BC")
    results["BC_invariant"] = metrics

    # Save results
    results_path = DATA_DIR / "mlp_baseline_results.json"
    # Convert numpy types
    clean = {}
    for k, v in results.items():
        clean[k] = {kk: float(vv) for kk, vv in v.items()}
    with open(results_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY (canonical SMILES)")
    print(f"{'='*60}")
    print(f"{'Group':<20} {'Spearman r_s':>12} {'MAE':>10}")
    print(f"{'-'*42}")
    for name in ["all", "A_bigrams", "B_morgan", "C_descriptors", "BC_invariant"]:
        r = results[name]
        print(f"{name:<20} {r['spearman_r']:>12.4f} {r['mae']:>10.4f}")


if __name__ == "__main__":
    main()
