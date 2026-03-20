"""
SMILES-augmented training — symmetry-consistent augmentation for molecular ML.

For each training sample, generate k random SMILES of the same molecule,
recompute SMILES-derived features (Group A), keep graph features (B, C) fixed.
This forces the model to learn representations invariant to SMILES ordering.

Augmentation levels: k ∈ {0, 1, 3, 7} (paralleling NPN augmentation sweep).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from data_prep import compute_smiles_bigrams_for_strings, generate_random_smiles
from train_mlp import MolMLP, load_config, evaluate

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = PROJECT_DIR / "checkpoints"


class AugmentedSMILESDataset(Dataset):
    """Dataset that includes SMILES-augmented copies of each molecule."""

    def __init__(self, features, targets, smiles, dim_A, bigram_vocab,
                 feature_mean, feature_std, n_augment=1):
        self.original_features = features
        self.targets = targets
        self.smiles = smiles
        self.dim_A = dim_A
        self.bigram_vocab = bigram_vocab
        self.feature_mean_A = feature_mean[:dim_A]
        self.feature_std_A = feature_std[:dim_A]
        self.n_augment = n_augment
        self.N = len(features)

        # Pre-generate augmented features for efficiency
        print(f"Pre-generating {n_augment} augmented copies per molecule...")
        self.all_features = [features.copy()]  # Original
        self.all_targets = [targets.copy()]

        for aug_i in range(n_augment):
            aug_features = features.copy()
            for j in range(self.N):
                smi = str(smiles[j])
                rand_smiles = generate_random_smiles(smi, n_random=1)
                bigram_feats = compute_smiles_bigrams_for_strings(rand_smiles, bigram_vocab)
                bigram_feats_norm = (bigram_feats[0] - self.feature_mean_A) / self.feature_std_A
                aug_features[j, :dim_A] = bigram_feats_norm
            self.all_features.append(aug_features)
            self.all_targets.append(targets.copy())

            if (aug_i + 1) % 1 == 0:
                print(f"  Augmentation copy {aug_i+1}/{n_augment} done")

        self.all_features = np.concatenate(self.all_features, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)
        print(f"Total training samples: {len(self.all_targets)} "
              f"({self.N} × {1 + n_augment})")

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, idx):
        return (torch.tensor(self.all_features[idx], dtype=torch.float32),
                torch.tensor(self.all_targets[idx], dtype=torch.float32))


def train_augmented(n_augment=1, seed=42):
    """Train with SMILES augmentation at specified level."""
    config = load_config()
    mc = config["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"SMILES AUGMENTED TRAINING (k={n_augment})")
    print(f"{'='*60}")

    # Load data
    data = np.load(DATA_DIR / "prepared.npz", allow_pickle=True)
    features = data["features"]
    targets = data["targets"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    smiles = data["smiles"]
    dim_A = int(data["dim_A"])
    bigram_vocab = list(data["bigram_vocab"])
    feature_mean = data["feature_mean"]
    feature_std = data["feature_std"]

    input_dim = features.shape[1]

    # Create augmented training set
    train_dataset = AugmentedSMILESDataset(
        features[train_idx], targets[train_idx], smiles[train_idx],
        dim_A, bigram_vocab, feature_mean, feature_std,
        n_augment=n_augment
    )
    train_loader = DataLoader(train_dataset, batch_size=mc["batch_size"], shuffle=True)

    # Validation and test (canonical, no augmentation)
    from torch.utils.data import TensorDataset
    val_loader = DataLoader(
        TensorDataset(torch.tensor(features[val_idx], dtype=torch.float32),
                       torch.tensor(targets[val_idx], dtype=torch.float32)),
        batch_size=mc["batch_size"])
    test_loader = DataLoader(
        TensorDataset(torch.tensor(features[test_idx], dtype=torch.float32),
                       torch.tensor(targets[test_idx], dtype=torch.float32)),
        batch_size=mc["batch_size"])

    # Model
    model = MolMLP(input_dim, mc["hidden_dims"], mc["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mc["lr"],
                                  weight_decay=mc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, mc["epochs"])
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"mlp_{n_augment}x_augment"
    ckpt_path = CKPT_DIR / f"{tag}_best.pt"
    best_val_rs = -1

    for epoch in range(mc["epochs"]):
        model.train()
        total_loss = 0
        n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        train_loss = total_loss / n

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics["spearman_r"] > best_val_rs:
            best_val_rs = val_metrics["spearman_r"]
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | loss={train_loss:.4f} "
                  f"| val_rs={val_metrics['spearman_r']:.4f} "
                  f"| best={best_val_rs:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    train_size = len(train_dataset)

    print(f"\nTest results ({n_augment}x augmentation, {train_size} samples):")
    print(f"  Spearman r_s = {test_metrics['spearman_r']:.4f}")
    print(f"  MAE          = {test_metrics['mae']:.4f}")

    return {
        "n_augment": n_augment,
        "train_size": train_size,
        "test_rs": float(test_metrics["spearman_r"]),
        "test_mae": float(test_metrics["mae"]),
        "tag": tag,
    }


def main():
    results = []
    for k in [1, 3, 7]:
        r = train_augmented(n_augment=k)
        results.append(r)

    # Save
    output_path = DATA_DIR / "augmentation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("AUGMENTATION SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Train size':>12} {'Test r_s':>10} {'MAE':>8}")
    print(f"{'-'*45}")
    for r in results:
        print(f"{r['n_augment']}x aug{'':<10} {r['train_size']:>12,} "
              f"{r['test_rs']:>10.4f} {r['test_mae']:>8.4f}")


if __name__ == "__main__":
    main()
